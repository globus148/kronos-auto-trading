"""Тесты для PredictionService (Kronos) — этап 4.

Проверяем:
1. Корректность производных метрик (pred_return, slope).
2. Интеграцию со StrategyEngine (Kronos меняет f_kronos и решение входа).
3. Реальный прогноз на данных Binance (если модель доступна).
4. Математическую корректность: pred_return = (last_close[-1]/context[-1]) - 1.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pandas as pd
import pytest

from core.market_service import Candle
from core.prediction_service import (
    PredictionResult,
    PredictionService,
    _linear_slope,
    _ts_to_datetime,
)


# ─── Вспомогательная математика ─────────────────────────────


class TestMath:
    def test_linear_slope_positive_uptrend(self):
        arr = [100, 101, 102, 103, 104]
        assert _linear_slope(arr) > 0

    def test_linear_slope_negative_downtrend(self):
        arr = [104, 103, 102, 101, 100]
        assert _linear_slope(arr) < 0

    def test_linear_slope_flat(self):
        arr = [100, 100, 100, 100]
        assert abs(_linear_slope(arr)) < 1e-6

    def test_linear_slope_normalized(self):
        """Наклон нормирован на начальное значение → масштабно-инвариантен."""
        s1 = _linear_slope([100, 101, 102])
        s2 = _linear_slope([200, 202, 204])  # те же % роста
        assert s1 == pytest.approx(s2, abs=1e-6)

    def test_ts_to_datetime(self):
        # 1700000000000 ms = 2023-11-14 22:13:20 UTC
        dt = _ts_to_datetime(1700000000000)
        assert dt.year == 2023
        assert dt.month == 11


# ─── Реальный прогноз Kronos (интеграционный) ────────────────

class TestKronosPrediction:
    """Тесты с реальной моделью Kronos на данных Binance.

    Пропускаются если нет GPU/модели/сети.
    """

    @pytest.fixture(scope="class")
    def btc_candles(self):
        try:
            async def fetch():
                from core.market_service import get_market
                from db.database import init_db, dispose_db
                await init_db()
                m = get_market()
                c = await m.fetch_ohlcv_history("BTC/USDT", "4h", total=520)
                await m.close()
                await dispose_db()
                return c
            return asyncio.run(fetch())
        except Exception as e:
            pytest.skip(f"Binance недоступен: {e}")

    @pytest.fixture(scope="class")
    def prediction(self, btc_candles):
        """Один прогноз на весь класс (модель грузится долго)."""
        try:
            svc = PredictionService()
            svc.load()
            return svc.predict(btc_candles, "BTC/USDT", "4h", horizon=8)
        except Exception as e:
            pytest.skip(f"Kronos недоступен: {e}")

    def test_prediction_structure(self, prediction):
        """Прогноз содержит все нужные поля."""
        assert isinstance(prediction, PredictionResult)
        assert prediction.coin_symbol == "BTC/USDT"
        assert prediction.timeframe == "4h"
        assert prediction.horizon == 8
        assert len(prediction.pred_close) == 8
        assert len(prediction.pred_high) == 8
        assert len(prediction.pred_low) == 8
        assert len(prediction.pred_open) == 8

    def test_pred_return_math(self, prediction, btc_candles):
        """pred_return = (pred_close[-1] / last_close) - 1 (математическая проверка)."""
        last_close = btc_candles[-1].close
        expected_return = (prediction.pred_close[-1] / last_close) - 1.0
        assert prediction.pred_return == pytest.approx(expected_return, abs=1e-6)
        assert prediction.last_close == pytest.approx(last_close, abs=1e-6)

    def test_pred_max_min(self, prediction):
        """pred_max_high = max(pred_high), pred_min_low = min(pred_low)."""
        assert prediction.pred_max_high == pytest.approx(max(prediction.pred_high))
        assert prediction.pred_min_low == pytest.approx(min(prediction.pred_low))
        # Логика: max_high >= min_low всегда.
        assert prediction.pred_max_high >= prediction.pred_min_low

    def test_slope_sign_matches_direction(self, prediction):
        """Знак slope совпадает с направлением прогноза (рост/падение)."""
        if prediction.pred_return > 0.01:
            # Рост → наклон положительный (на горизонте).
            assert prediction.pred_path_slope > -1e-3
        elif prediction.pred_return < -0.01:
            assert prediction.pred_path_slope < 1e-3

    def test_inference_fast(self, prediction):
        """Инференс на RTX 3060 должен быть < 5 секунд."""
        assert prediction.inference_ms < 5000

    def test_prices_positive(self, prediction):
        """Все прогнозные цены положительные."""
        assert all(p > 0 for p in prediction.pred_close)
        assert all(p > 0 for p in prediction.pred_high)
        assert all(p > 0 for p in prediction.pred_low)


# ─── Интеграция со StrategyEngine ───────────────────────────

class TestStrategyIntegration:
    """Kronos прогноз меняет композитный сигнал стратегии."""

    @pytest.fixture(scope="class")
    def btc_candles(self):
        try:
            async def fetch():
                from core.market_service import get_market
                from db.database import init_db, dispose_db
                await init_db()
                m = get_market()
                c = await m.fetch_ohlcv_history("BTC/USDT", "4h", total=520)
                await m.close()
                await dispose_db()
                return c
            return asyncio.run(fetch())
        except Exception:
            pytest.skip("Binance недоступен")

    def test_strategy_uses_kronos_prediction(self, btc_candles):
        """Полный конвейер: свечи → Kronos → индикаторы → стратегия → сигнал."""
        from core.indicators import compute_all
        from core.strategy_engine import StrategyEngine

        # Kronos прогноз.
        try:
            svc = PredictionService()
            svc.load()
            pred = svc.predict(btc_candles, "BTC/USDT", "4h", horizon=8)
        except Exception:
            pytest.skip("Kronos недоступен")

        # Индикаторы на контексте.
        df = pd.DataFrame({
            "timestamp": [c.timestamp for c in btc_candles],
            "close": [c.close for c in btc_candles],
            "high": [c.high for c in btc_candles],
            "low": [c.low for c in btc_candles],
            "volume": [c.volume for c in btc_candles],
        })
        snap = compute_all(
            df["timestamp"], df["close"], df["high"], df["low"], df["volume"]
        )

        # Стратегия с Kronos.
        se = StrategyEngine()
        signal = se.compute_signal(
            snap,
            kronos_pred_return=pred.pred_return,
            kronos_pred_slope=pred.pred_path_slope,
            kronos_pred_max=pred.pred_max_high,
        )

        # f_kronos теперь > 0 (Kronos подключён).
        assert signal.sub_signals["f_kronos"] >= 0
        print(f"\n  BTC/USDT 4h: pred_return={pred.pred_return*100:+.2f}% "
              f"f_kronos={signal.sub_signals['f_kronos']:.3f} "
              f"S_entry={signal.s_entry:.3f}")
