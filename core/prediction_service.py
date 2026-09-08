"""PredictionService — интеграция Kronos foundation model (план 4.2).

Загружает Kronos (predictor + tokenizer) локально на GPU (RTX 3060) и
прогнозирует N свечей вперёд на основе OHLCV.

Ключевые особенности:
    - Ленивая загрузка: модель грузится только при первом predict()
    - Кэш прогнозов по (coin, tf, last_timestamp) — повторные запросы дешевле
    - Нормализация/денормализация выполняется внутри KronosPredictor
    - Возвращает PredictionResult с метриками для StrategyEngine

План: раздел 4.2 (прогноз Kronos как ядро feature vector).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import settings
from core.market_service import Candle

logger = logging.getLogger("trading.predict")

# Метрики по умолчанию когда модель не загружена.
_NO_MODEL_PLACEHOLDER = None


@dataclass
class PredictionResult:
    """Результат прогноза Kronos на N свечей вперёд.

    Все поля нужны StrategyEngine для feature vector (план 4.2).
    """
    coin_symbol: str
    timeframe: str
    predicted_at: int          # ms epoch момента прогноза
    last_close: float          # close последней свечи контекста
    horizon: int               # N свечей вперёд
    pred_close: list[float]    # прогноз close по каждой свече
    pred_high: list[float]
    pred_low: list[float]
    pred_open: list[float]
    pred_volume: list[float]
    # Производные метрики (план 4.2):
    pred_return: float         # доходность за горизонт
    pred_max_high: float       # максимальный high в прогнозе
    pred_min_low: float        # минимальный low в прогнозе
    pred_path_slope: float     # наклон линейной регрессии по close
    inference_ms: float        # время инференса в мс


class PredictionService:
    """Сервис прогноза Kronos с ленивой загрузкой и кэшем.

    Использование::

        svc = PredictionService()
        await svc.load()                       # опционально, иначе lazy
        result = svc.predict(candles, "4h")    # sync (модель на GPU)
    """

    def __init__(self) -> None:
        self._predictor = _NO_MODEL_PLACEHOLDER
        self._cache: dict[tuple[str, str, int], PredictionResult] = {}
        # Ограничиваем кэш по размеру (FIFO через OrderedDict-логику).
        self._cache_max = 100
        # Отдельный кэш для мини-бэктеста (TTL 30 мин).
        self._backtest_cache: dict[tuple, dict] = {}

    # ─── Загрузка модели ─────────────────────────────────────

    def load(self) -> None:
        """Загрузить Kronos + tokenizer с HuggingFace (один раз на процесс).

        Синхронная операция (PyTorch). Грузится в CUDA если доступна.
        """
        if self._predictor is not None:
            return

        import torch
        from kronos import Kronos, KronosPredictor, KronosTokenizer

        device = settings.kronos_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA недоступна — переключение на CPU")
            device = "cpu"

        t0 = time.time()
        logger.info(
            "Загрузка Kronos: model=%s tokenizer=%s device=%s",
            settings.kronos_model, settings.kronos_tokenizer, device,
        )
        tokenizer = KronosTokenizer.from_pretrained(settings.kronos_tokenizer)
        model = Kronos.from_pretrained(settings.kronos_model)
        self._predictor = KronosPredictor(
            model, tokenizer,
            max_context=settings.data_kronos_context,
        )
        elapsed = time.time() - t0
        logger.info("Kronos загружена за %.1fс (device=%s)", elapsed, device)

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    # ─── Прогноз ─────────────────────────────────────────────

    def predict(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
        horizon: int | None = None,
    ) -> PredictionResult:
        """Спрогнозировать `horizon` свечей вперёд.

        Args:
            candles: история OHLCV (хронологический порядок), хотя бы
                     data_kronos_context свечей.
            symbol: символ монеты ("BTC/USDT").
            timeframe: таймфрейм ("1h", "4h").
            horizon: сколько свечей вперёд (по умолч. settings.kronos_pred_len).
        """
        if not self.is_loaded:
            self.load()

        horizon = horizon or settings.kronos_pred_len
        if len(candles) < settings.data_kronos_context:
            raise ValueError(
                f"Недостаточно свечей для Kronos: {len(candles)} < "
                f"{settings.data_kronos_context}"
            )

        # Кэш: ключ = (symbol, tf, timestamp последней свечи контекста).
        last_ts = candles[-1].timestamp
        cache_key = (symbol, timeframe, last_ts, horizon)
        if cache_key in self._cache:
            logger.debug("Прогноз из кэша: %s %s", symbol, timeframe)
            return self._cache[cache_key]

        result = self._run_inference(candles, symbol, timeframe, horizon)

        # Сохраняем в кэш (с ограничением размера).
        self._cache[cache_key] = result
        if len(self._cache) > self._cache_max:
            # Удаляем самый старый ключ.
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return result

    def _run_inference(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
        horizon: int,
    ) -> PredictionResult:
        """Непосредственно вызов KronosPredictor.predict."""
        # Берём последние kronos_context свечей как контекст.
        ctx_size = settings.data_kronos_context
        context = candles[-ctx_size:]

        # Готовим DataFrame в формате Kronos.
        # Kronos.calc_time_stamps вызывает .dt.minute → нужны pd.Series[datetime].
        x_timestamps = pd.Series([_ts_to_datetime(c.timestamp) for c in context])
        # Будущие таймстампы: генерируем по интервалу ТФ.
        tf_ms = _tf_ms(timeframe)
        last_ts = context[-1].timestamp
        y_timestamps = pd.Series([
            _ts_to_datetime(last_ts + (i + 1) * tf_ms) for i in range(horizon)
        ])

        df = pd.DataFrame({
            "open": [c.open for c in context],
            "high": [c.high for c in context],
            "low": [c.low for c in context],
            "close": [c.close for c in context],
            "volume": [c.volume for c in context],
            "amount": [c.volume * c.close for c in context],
        })

        t0 = time.time()
        pred_df = self._predictor.predict(
            df, x_timestamps, y_timestamps, horizon,
            T=settings.kronos_temperature,
            top_p=settings.kronos_top_p,
            sample_count=settings.kronos_sample_count,
            verbose=False,
        )
        inference_ms = (time.time() - t0) * 1000

        # Извлекаем производные метрики (план 4.2).
        pred_close_arr = pred_df["close"].values
        last_close = context[-1].close
        pred_return = float((pred_close_arr[-1] / last_close) - 1.0) if last_close > 0 else 0.0
        pred_max_high = float(pred_df["high"].max())
        pred_min_low = float(pred_df["low"].min())
        pred_path_slope = _linear_slope(pred_close_arr)

        result = PredictionResult(
            coin_symbol=symbol, timeframe=timeframe,
            predicted_at=_now_ms(), last_close=float(last_close),
            horizon=horizon,
            pred_close=[float(v) for v in pred_close_arr],
            pred_high=[float(v) for v in pred_df["high"].values],
            pred_low=[float(v) for v in pred_df["low"].values],
            pred_open=[float(v) for v in pred_df["open"].values],
            pred_volume=[float(v) for v in pred_df["volume"].values],
            pred_return=pred_return,
            pred_max_high=pred_max_high,
            pred_min_low=pred_min_low,
            pred_path_slope=pred_path_slope,
            inference_ms=inference_ms,
        )
        logger.info(
            "Прогноз %s %s: return=%.2f%% max_high=%.2f slope=%.5f (%.0f мс)",
            symbol, timeframe, pred_return * 100, pred_max_high,
            pred_path_slope, inference_ms,
        )
        return result

    def clear_cache(self) -> None:
        self._cache.clear()
        self._backtest_cache.clear()

    # ─── Мини-бэктест точности (план: визуализация на графике) ────

    def mini_backtest(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
        steps: int = 16,
    ) -> list[dict]:
        """Мгновенный мини-бэктест точности Kronos на последних `steps` свечах.

        Для каждой из последних `steps` свечей:
        - берём контекст ДО неё (последние kronos_context свечей),
        - прогнозируем 1 свечу вперёд,
        - сравниваем направление (close растёт/падает) с реальным.

        Возвращает список (хронологически)::
            [{"timestamp", "predicted_up": bool, "actual_up": bool, "correct": bool}, ...]

        Кеш 30 минут по ключу (symbol, tf, last_ts, steps) — повторные загрузки
        страницы мгновенны. Первый прогон: ~steps инференсов.
        """
        if not self.is_loaded:
            self.load()

        ctx_size = settings.data_kronos_context
        # Нужно минимум ctx_size свечей до первой проверяемой + steps проверяемых.
        min_needed = ctx_size + steps
        if len(candles) < min_needed:
            logger.warning(
                "mini_backtest %s: мало свечей %d < %d",
                symbol, len(candles), min_needed,
            )
            return []

        last_ts = candles[-1].timestamp
        cache_key = (symbol, timeframe, last_ts, steps)
        cached = self._backtest_cache.get(cache_key)
        if cached is not None:
            logger.debug("mini_backtest из кэша: %s %s", symbol, timeframe)
            return cached["results"]

        results: list[dict] = []
        for i in range(steps):
            # Индекс проверяемой свечи: идём с конца назад.
            # candles[-(steps - i)] — проверяемая свеча.
            # Контекст: всё до неё (берём последние ctx_size).
            check_idx = len(candles) - steps + i
            context = candles[max(0, check_idx - ctx_size):check_idx]
            actual_candle = candles[check_idx]
            if len(context) < ctx_size:
                continue

            # Прогноз 1 свечи вперёд (кешируется внутри predict).
            pred = self.predict(context, symbol, timeframe, horizon=1)
            predicted_close = pred.pred_close[0]
            context_last_close = context[-1].close
            actual_close = actual_candle.close

            predicted_up = predicted_close > context_last_close
            actual_up = actual_close > context_last_close
            correct = predicted_up == actual_up

            results.append({
                "timestamp": actual_candle.timestamp,
                "predicted_up": predicted_up,
                "actual_up": actual_up,
                "correct": correct,
            })

        # Сохраняем в кэш с TTL 30 мин.
        self._backtest_cache[cache_key] = {
            "results": results,
            "cached_at": time.time(),
        }
        self._evict_backtest_cache()

        correct_count = sum(1 for r in results if r["correct"])
        logger.info(
            "mini_backtest %s %s: точность %d/%d (%.0f%%)",
            symbol, timeframe, correct_count, len(results),
            (correct_count / len(results) * 100) if results else 0,
        )
        return results

    def _evict_backtest_cache(self) -> None:
        """Очистка кэша мини-бэктеста: TTL 30 мин + лимит размера."""
        now = time.time()
        ttl = 1800  # 30 минут
        # Удаляем просроченные.
        expired = [k for k, v in self._backtest_cache.items()
                   if now - v["cached_at"] > ttl]
        for k in expired:
            del self._backtest_cache[k]
        # Лимит размера (FIFO).
        while len(self._backtest_cache) > 20:
            oldest = next(iter(self._backtest_cache))
            del self._backtest_cache[oldest]


# ─── Вспомогательные ─────────────────────────────────────────


def _ts_to_datetime(ts_ms: int) -> datetime:
    """ms epoch → datetime (UTC). Kronos требует datetime для timestamp-ов."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _tf_ms(timeframe: str) -> int:
    import re
    m = re.match(r"^(\d+)([smhdw])$", timeframe)
    if not m:
        raise ValueError(f"Неподдерживаемый ТФ: {timeframe}")
    n, unit = int(m.group(1)), m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return n * mult * 1000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _linear_slope(arr: np.ndarray | list[float]) -> float:
    """Нормированный наклон линейной регрессии по ряду close.

    Положительный → восходящий прогноз, отрицательный → нисходящий.
    Нормировка на начальное значение делает метрику масштабно-инвариантной.
    """
    y = np.asarray(arr, dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    # y = a*x + b; slope = a.
    a = np.polyfit(x, y, 1)[0]
    base = y[0] if y[0] != 0 else 1e-8
    return float(a / base)


# ─── Синглтон ────────────────────────────────────────────────

_predictor_svc: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _predictor_svc
    if _predictor_svc is None:
        _predictor_svc = PredictionService()
    return _predictor_svc
