"""TradingLoop — живой торговый цикл на APScheduler (план, раздел 8).

Оркестратор: на закрытии каждой свечи ТФ прогоняет полный конвейер
для каждой включённой монеты:

    1. DataLifecycleManager.update_latest() — fetch + prune (план 6.5)
    2. Indicators.compute_all() — снапшот индикаторов
    3. PredictionService.predict() — прогноз Kronos (кэш)
    4. StrategyEngine.compute_signal() — S_entry / S_exit
    5. PortfolioEngine: проверка exit по открытым позициям (план 4.4)
    6. PortfolioEngine: вход если авто-режим ВКЛ и S_entry ≥ порога (план 4.3)
    7. PortfolioEngine.record_equity() → push в WebSocket (этап 10)

Конфигурация таймфреймов: APScheduler cron/jobs на закрытие свечи.
Для прототипа также есть метод run_once() для ручного прогона (тесты, UI-кнопка).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from config import settings
from core.data_lifecycle import get_dlm
from core.indicators import compute_all
from core.market_service import get_market
from core.portfolio_engine import get_portfolio
from core.prediction_service import get_prediction_service
from core.risk_manager import RiskManager
from core.strategy_config import refresh_snapshot
from core.strategy_engine import StrategyEngine
from core.mode_manager import get_active_mode
from core.news_service import get_news_service
from db.database import get_session
from db.models import (
    AutoConfig,
    Coin,
    Position,
    PositionSide,
    PositionStatus,
    Session,
    SessionMode,
    TradeReason,
)

logger = logging.getLogger("trading.loop")

# Колбэк для WebSocket-push (этап 10). None по умолчанию.
OnTickCallback = Callable[[dict], None]


@dataclass
class TickResult:
    """Результат одного тика цикла по монете (для логов и UI)."""
    symbol: str
    timeframe: str
    s_entry: float
    s_exit: float
    action: str          # "hold" | "opened" | "opened:short" | "closed:reason" | "skip"
    detail: str = ""
    s_entry_short: float = 0.0
    direction: str = "up"   # "up" (прогноз роста) | "down" (прогноз падения)
    close: float = 0.0       # текущая цена (для сбора в equity)


class TradingLoop:
    """Оркестратор торгового цикла.

    Один экземпляр на приложение. Управляет scheduler'ом и состоянием
    авто-торговли по монетам.
    """

    def __init__(
        self,
        strategy: StrategyEngine | None = None,
        risk_mgr: RiskManager | None = None,
        on_tick: OnTickCallback | None = None,
    ) -> None:
        self.strategy = strategy or StrategyEngine(risk_mgr or RiskManager())
        self.scheduler = AsyncIOScheduler()
        self.on_tick = on_tick
        # Какая монета в каком ТФ торгуется в авто-режиме.
        # key = "BTC/USDT:4h", value = bool (auto включён).
        self._auto_enabled: dict[str, bool] = {}
        # Кеш последнего сигнала по каждой паре/ТФ для live-отображения (без GPU).
        # key = "BTC/USDT:4h", value = dict с s_entry, sub_signals, close и т.д.
        self._last_signals: dict[str, dict] = {}
        # Кеш timestamp последней обработанной свечи для каждой пары/ТФ.
        # Используется чтобы отличить «новая закрытая свеча» от polling-refresh:
        # bars_held позиций инкрементируется только когда сменилась свеча, а не
        # на каждом фоновом прогоне (иначе time_stop срабатывает в 6 раз быстрее).
        # key = "BTC/USDT:4h", value = int (ms timestamp последней свечи).
        self._last_bar_ts: dict[str, int] = {}
        # Timestamp последней записи equity, для throttling (не чаще раз в 10с,
        # чтобы не было дублей при пересечении cron'ов).
        self._last_equity_ts: float = 0.0
        # Флаг фоновой перегонки прогнозов: чтобы cron (15 мин) и ручной /recalc
        # не запустились одновременно (получили бы дубль инференсов на GPU).
        self._refreshing: bool = False
        self._started = False
        # Timestamp старта для uptime в health endpoint.
        self._start_time: datetime = datetime.utcnow()

    # ─── Управление авто-режимом ─────────────────────────────

    def set_auto(self, symbol: str, timeframe: str, enabled: bool) -> None:
        """Включить/выключить авто-торговлю для пары/ТФ.

        Сохраняет в БД (AutoConfig) — состояние переживает рестарт процесса.
        In-memory dict обновляется мгновенно для быстрых чтений в is_auto().
        """
        key = f"{symbol}:{timeframe}"
        self._auto_enabled[key] = enabled
        logger.info("Авто-режим %s %s: %s", symbol, timeframe,
                    "ВКЛ" if enabled else "ВЫКЛ")
        # Асинхронно сохраняем в БД (не блокируем текущий корутин).
        try:
            asyncio.ensure_future(self._persist_auto(key, enabled))
        except RuntimeError:
            # Нет event loop (вызов синхронно из тестов) — игнорируем.
            pass

    @staticmethod
    async def _persist_auto(pair: str, enabled: bool) -> None:
        """Сохранить флаг авто-режима в БД (upsert)."""
        try:
            async with get_session() as db:
                existing = (await db.execute(
                    select(AutoConfig).where(AutoConfig.pair == pair)
                )).scalar_one_or_none()
                if existing:
                    existing.enabled = enabled
                else:
                    db.add(AutoConfig(pair=pair, enabled=enabled))
                await db.commit()
        except Exception as exc:
            logger.warning("Не удалось сохранить авто-режим %s в БД: %s", pair, exc)

    async def _restore_auto_state(self) -> None:
        """Загрузить сохранённые авто-флаги из БД при старте.

        Вызывается в start() — восстанавливает состояние после рестарта,
        чтобы авто-торговля не сбрасывалась в OFF при перезапуске сервера
        (важно для туннелей: Windows sleep / обрыв ngrok не теряют авто).
        """
        try:
            async with get_session() as db:
                rows = (await db.execute(
                    select(AutoConfig).where(AutoConfig.enabled.is_(True))
                )).scalars().all()
                for row in rows:
                    self._auto_enabled[row.pair] = True
                if rows:
                    logger.info("Восстановлены авто-флаги для %d пар: %s",
                                len(rows), [r.pair for r in rows])
        except Exception as exc:
            logger.warning("Не удалось загрузить авто-флаги из БД: %s", exc)

    def is_auto(self, symbol: str, timeframe: str) -> bool:
        return self._auto_enabled.get(f"{symbol}:{timeframe}", False)

    def get_last_signal(self, symbol: str, timeframe: str) -> dict | None:
        """Последний посчитанный сигнал для пары/ТФ (для live-отображения).

        Возвращает dict или None, если тик ещё не выполнялся. Чтение из кеша —
        мгновенно, без обращения к GPU/Binance.
        """
        return self._last_signals.get(f"{symbol}:{timeframe}")

    def get_health(self) -> dict:
        """Состояние планировщика для /api/health (мониторинг туннеля/простоя)."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds() if self._started else 0
        auto_pairs = [k for k, v in self._auto_enabled.items() if v]
        return {
            "started": self._started,
            "scheduler_running": self.scheduler.running if self._started else False,
            "uptime_seconds": int(uptime),
            "auto_pairs": auto_pairs,
            "jobs": [
                {"id": job.id, "next_run": str(job.next_run_time)}
                for job in self.scheduler.get_jobs()
            ] if self._started else [],
        }

    # ─── Жизненный цикл scheduler'а ──────────────────────────

    async def start(self) -> None:
        """Запустить планировщик: cron-задачи на закрытие свечи каждого ТФ."""
        if self._started:
            return
        # Восстановить сохранённые авто-флаги из БД (переживают рестарт).
        await self._restore_auto_state()
        # Для каждого ТФ — отдельная cron-задача, срабатывающая на закрытии свечи.
        # 1h: на 1-й минуте каждого часа. 4h: на 1-й минуте каждые 4ч (0,4,8,12,16,20).
        tf_cron = {
            "1h": {"minute": 1},
            "4h": {"minute": 1, "hour": "0,4,8,12,16,20"},
        }
        for tf in settings.default_timeframes:
            cron_kwargs = tf_cron.get(tf, {"minute": 1})
            self.scheduler.add_job(
                self._tick_all_coins,
                trigger=CronTrigger(**cron_kwargs),
                id=f"tick_{tf}",
                args=[tf],
                replace_existing=True,
                # 5 минут — если event loop был занят (GPU-инференс, Binance-таймаут)
                # и job пропустила слот, она всё равно выполнится. Без этого дефолт
                # APScheduler = 1 сек → job молча пропускается.
                misfire_grace_time=300,
            )
            logger.info("Запланирован цикл для ТФ %s: %s", tf, cron_kwargs)
        # Фоновое обновление сигналов/прогнозов каждые 15 минут — чтобы дашборд
        # всегда показывал свежие вероятности роста без ожидания закрытия свечи.
        # Использует кеш Kronos по (symbol, tf, last_ts): повторные инференсы в
        # пределах одной свечи не делаются (дешево, пока свеча не сменилась).
        self.scheduler.add_job(
            self.refresh_all_predictions,
            trigger=CronTrigger(minute="*/15"),
            id="refresh_predictions",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info("Запланирован фоновый прогон прогнозов: каждые 15 мин")
        self.scheduler.start()
        self._started = True
        self._start_time = datetime.utcnow()
        logger.info("TradingLoop запущен")

    async def stop(self) -> None:
        """Остановить планировщик."""
        if self._started:
            self.scheduler.shutdown(wait=False)
            self._started = False
            logger.info("TradingLoop остановлен")
        # Закрыть HTTP-сессию новостного сервиса (Gemini).
        try:
            from core.news_service import _news_svc
            if _news_svc is not None:
                await _news_svc.close()
        except Exception:
            pass

    async def _tick_all_coins(self, timeframe: str) -> None:
        """Прогнать цикл по всем включённым монетам для заданного ТФ.

        После прохода по всем монетам записывает equity ОДИН раз со
        собранными ценами (а не внутри run_once по одной монете).
        """
        async with get_session() as db:
            stmt = select(Coin).where(Coin.enabled == True)
            coins = (await db.execute(stmt)).scalars().all()
        # Собираем последние цены со всех монет (run_once возвращает snap.close).
        collected_prices: dict[int, float] = {}
        for coin in coins:
            try:
                tick = await self.run_once(coin.symbol, timeframe)
                if tick and tick.close:
                    collected_prices[coin.id] = tick.close
            except Exception as e:
                logger.exception("Ошибка тика %s %s: %s", coin.symbol, timeframe, e)
        # Записываем equity ОДИН раз за весь тик со всеми ценами.
        await self._record_global_equity(collected_prices)

    async def _record_global_equity(self, prices: dict[int, float]) -> None:
        """Записать equity-точку один раз со собранными ценами всех монет.

        Вызывается после прохода по всем монетам в _tick_all_coins /
        refresh_all_predictions. Throttle: не чаще раза в 10 секунд, чтобы
        при пересечении cron'ов не было дублей.
        """
        if not prices:
            return
        import time
        now = time.monotonic()
        if now - self._last_equity_ts < 10.0:
            return
        self._last_equity_ts = now
        portfolio = get_portfolio()
        mode = await get_active_mode()
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, mode)
            await portfolio.record_equity(db, session, prices=prices)
            await db.commit()

    # ─── Фоновый прогон прогнозов для дашборда ─────────────

    @property
    def is_refreshing(self) -> bool:
        """Идёт ли сейчас фоновая (или ручная) перегонка прогнозов."""
        return self._refreshing

    async def refresh_all_predictions(self) -> None:
        """Пересчитать сигнал/прогноз по всем монетам × ТФ для дашборда.

        Запускается cron'ом каждые 15 минут и кнопкой «Пересчитать» в UI.
        Гарантирует, что _last_signals (win_prob, pred_return для каждого ТФ)
        всегда свежий, не дожидаясь закрытия свечи.

        Использует кеш Kronos: в пределах одной свечи инференс для пары/ТФ
        выполняется один раз — повторные прогоны читают кеш (мгновенно).

        Equity записывается ОДИН раз после прохода всех монет.
        """
        if self._refreshing:
            logger.debug("refresh_all_predictions: уже идёт, пропускаем")
            return
        self._refreshing = True
        try:
            async with get_session() as db:
                coins = (await db.execute(
                    select(Coin).where(Coin.enabled == True)
                )).scalars().all()
            total = len(coins) * len(settings.default_timeframes)
            logger.info("Фоновый прогон прогнозов: %d монет × %d ТФ = %d",
                        len(coins), len(settings.default_timeframes), total)
            done = 0
            collected_prices: dict[int, float] = {}
            for coin in coins:
                for tf in settings.default_timeframes:
                    try:
                        tick = await self.run_once(coin.symbol, tf)
                        if tick and tick.close:
                            collected_prices[coin.id] = tick.close
                        done += 1
                    except Exception as e:
                        logger.exception("Фоновый прогон %s %s: %s",
                                         coin.symbol, tf, e)
            # Записываем equity ОДИН раз за весь прогон.
            await self._record_global_equity(collected_prices)
            logger.info("Фоновый прогон завершён: %d/%d обновлено", done, total)
        finally:
            self._refreshing = False

    # ─── Основной прогон одной монеты ────────────────────────

    async def _try_enter_long(
        self, portfolio, db, session, coin, snap, signal,
        timeframe, pred_return, pred_max_high,
    ) -> tuple[bool, str, str, str]:
        """Попытка входа в LONG. Возвращает (вошёл, action, detail, reject_reason)."""
        should_enter, sizing, reject_reason = self.strategy.evaluate_entry(
            snap, signal.s_entry, signal.sub_signals,
            equity=session.cash,
            cash=session.cash,
            kronos_pred_return=pred_return,
            kronos_pred_max=pred_max_high,
        )
        if not should_enter or sizing is None:
            logger.info(
                "LONG отказ %s %s: s_entry=%.3f | %s",
                coin.symbol if hasattr(coin, 'symbol') else symbol,
                timeframe, signal.s_entry, reject_reason,
            )
            return False, "", "", reject_reason
        result = await portfolio.open_position(
            db, session, coin, sizing, timeframe,
            TradeReason.entry_signal,
        )
        if result is None:
            return False, "", "", "ошибка открытия позиции"
        return True, "opened", f"qty={sizing.qty:.6f}", ""

    async def _try_enter_short(
        self, portfolio, db, session, coin, snap, signal,
        timeframe, pred_return, pred_min_low,
    ) -> tuple[bool, str, str, str]:
        """Попытка входа в SHORT. Возвращает (вошёл, action, detail, reject_reason)."""
        should_short, sizing, reject_reason = self.strategy.evaluate_short_entry(
            snap, signal.s_entry_short, signal.sub_signals_short,
            equity=session.cash,
            cash=session.cash,
            kronos_pred_return=pred_return,
            kronos_pred_min=pred_min_low,
        )
        if not should_short or sizing is None:
            logger.info(
                "SHORT отказ %s %s: s_entry_short=%.3f | %s",
                coin.symbol if hasattr(coin, 'symbol') else symbol,
                timeframe, signal.s_entry_short, reject_reason,
            )
            return False, "", "", reject_reason
        result = await portfolio.open_short_position(
            db, session, coin, sizing, timeframe,
            TradeReason.entry_signal,
        )
        if result is None:
            return False, "", "", "ошибка открытия позиции"
        return True, "opened:short", f"qty={sizing.qty:.6f}", ""

    async def run_once(self, symbol: str, timeframe: str) -> TickResult:
        """Полный конвейер для одной монеты/ТФ.

        Используется и scheduler'ом, и UI-кнопкой «обновить», и тестами.
        """
        market = get_market()
        dlm = get_dlm()
        portfolio = get_portfolio()
        pred_svc = get_prediction_service()

        # Обновляем снапшот редактируемых порогов (меняются через /settings).
        await refresh_snapshot()

        async with get_session() as db:
            # 1. Монета и сессия.
            coin = (await db.execute(
                select(Coin).where(Coin.symbol == symbol)
            )).scalar_one_or_none()
            if coin is None:
                self._last_signals[f"{symbol}:{timeframe}"] = {
                    "symbol": symbol, "tf": timeframe,
                    "s_entry": 0, "s_entry_short": 0, "s_exit": 0,
                    "sub_signals": {}, "sub_signals_short": {},
                    "close": 0, "action": "skip",
                    "pred_return": None, "pred_path_slope": None,
                    "direction": "up",
                    "win_prob": 0, "updated_at": datetime.utcnow().isoformat(),
                }
                return TickResult(symbol, timeframe, 0, 0, "skip", "монета не найдена")

            session = await portfolio.get_or_create_session(db, await get_active_mode())

            # 2. Обновить данные (план 6.5): fetch последней свечи + prune.
            # Если данных совсем мало (первый запуск) — ensure_history сделает cold start.
            await dlm.ensure_history(db, coin, timeframe)

            # 3. Получить последние свечи для индикаторов и Kronos.
            candles = await dlm.get_candles(
                db, coin.id, timeframe,
                limit=max(settings.data_kronos_context, settings.data_indicator_warmup) + 5,
            )
            if len(candles) < settings.data_indicator_warmup:
                # Кэшируем частичный сигнал чтобы UI показал хотя бы цену.
                last_close = candles[-1].close if candles else 0
                self._last_signals[f"{symbol}:{timeframe}"] = {
                    "symbol": symbol, "tf": timeframe,
                    "s_entry": 0, "s_entry_short": 0, "s_exit": 0,
                    "sub_signals": {"f_trend": 0, "f_kronos": 0, "f_momentum": 0, "f_rsi": 0, "f_pullback": 0},
                    "sub_signals_short": {"f_trend": 0, "f_kronos": 0, "f_momentum": 0, "f_rsi": 0, "f_pullback": 0},
                    "close": last_close, "action": "skip",
                    "pred_return": None, "pred_path_slope": None,
                    "direction": "up",
                    "win_prob": 0, "updated_at": datetime.utcnow().isoformat(),
                }
                return TickResult(symbol, timeframe, 0, 0, "skip",
                                  f"мало данных: {len(candles)}")

            # 4. Индикаторы.
            df = pd.DataFrame({
                "timestamp": [c.timestamp for c in candles],
                "close": [c.close for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "volume": [c.volume for c in candles],
            })
            snap = compute_all(
                df["timestamp"], df["close"], df["high"], df["low"], df["volume"]
            )

            # 5. Kronos прогноз (в отдельном потоке — модель синхронная/GPU).
            pred = None
            if pred_svc.is_loaded:
                pred = await asyncio.to_thread(
                    lambda: pred_svc.predict(candles, symbol, timeframe)
                )
            # Если модель не загружена — pred остаётся None, стратегия работает
            # на индикаторах (f_kronos=0).

            # 5.5 Новости: сентимент через LLM (OpenRouter или Gemini).
            # Если ключа нет / ошибка — news_result = None → f_news = 0.5 (нейтрально).
            news_result = None
            # Проверяем ключ текущего провайдера (не gemini_key для openrouter).
            news_key_ok = (
                settings.news_enabled and
                ((settings.news_provider == "groq" and settings.groq_api_key)
                 or (settings.news_provider == "openrouter" and settings.openrouter_api_key)
                 or (settings.news_provider == "gemini" and settings.gemini_api_key))
            )
            if news_key_ok:
                try:
                    news_svc = get_news_service()
                    news_result = await news_svc.get_sentiment(symbol)
                except Exception as exc:
                    # Логируем коротко (не полный traceback) — новости не критичны,
                    # стратегия работает и без них (f_news = 0.5 нейтрально).
                    logger.warning("Новости %s недоступны: %s", symbol, exc)
            news_sentiment = news_result.sentiment if news_result else None

            # 6. Сигнал стратегии (long + short).
            signal = self.strategy.compute_signal(
                snap,
                kronos_pred_return=pred.pred_return if pred else None,
                kronos_pred_slope=pred.pred_path_slope if pred else None,
                kronos_pred_max=pred.pred_max_high if pred else None,
                kronos_pred_min=pred.pred_min_low if pred else None,
                news_sentiment=news_sentiment,
            )

            # Направление прогноза для UI (up/down) — ЕДИНАЯ механика с дашбордом:
            # по pred_return (прогноз Kronos). pred_return > 0 → рост, иначе → падение.
            # Это совпадает с тем, что рисует дашборд (applyBar: up = predReturn > 0).
            pred_ret = pred.pred_return if pred else None
            pred_slope = pred.pred_path_slope if pred else None
            if pred_ret is not None:
                direction = "down" if pred_ret <= 0 else "up"
            elif pred_slope is not None:
                # Нет pred_return, но есть наклон пути прогноза.
                direction = "down" if pred_slope < 0 else "up"
            else:
                # Kronos не загружен — fallback на EMA20 vs EMA50.
                direction = "down" if snap.ema20 < snap.ema50 else "up"

            # 7. Проверить exit по открытым позициям (план 4.4), с учётом side.
            # С pyramiding может быть несколько позиций по одной монете — проверяем все.
            # bars_held инкрементируется ТОЛЬКО при смене свечи (новый закрытый бар),
            # а не на каждом фоновом прогоне — иначе time_stop срабатывает в разы быстрее.
            key_pair = f"{symbol}:{timeframe}"
            current_bar_ts = candles[-1].timestamp
            prev_bar_ts = self._last_bar_ts.get(key_pair)
            # Первое наблюдение (после старта/рестарта) НЕ считаем новой свечой —
            # просто запоминаем baseline, иначе каждый рестарт даёт ложный +1.
            new_bar_closed = prev_bar_ts is not None and current_bar_ts > prev_bar_ts
            self._last_bar_ts[key_pair] = current_bar_ts

            action = "hold"
            detail = ""
            # Берём открытые позиции по монете, но обрабатываем только те,
            # чей ТФ совпадает с текущим тиком. Позиция 4h не должна
            # инкрементировать bars_held при обработке 1h-тика (иначе она
            # стареет в 5 раз быстрее и срабатывает time_stop за часы, а не дни).
            all_positions = await portfolio.get_open_positions_by_coin(db, session, coin)
            open_positions = [p for p in all_positions if p.tf == timeframe]
            for open_pos in open_positions:
                reason = self.strategy.evaluate_exit(
                    snap,
                    position_price=open_pos.entry_price,
                    stop_price=open_pos.stop_price,
                    target_price=open_pos.target_price,
                    bars_held=open_pos.bars_held,
                    trailing_stop=open_pos.trailing_stop,
                    kronos_pred_return=pred.pred_return if pred else None,
                    kronos_pred_slope=pred.pred_path_slope if pred else None,
                    side=open_pos.side.value if hasattr(open_pos.side, "value") else str(open_pos.side),
                )
                if reason is not None:
                    closed = await portfolio.close_position(
                        db, session, open_pos, coin, snap.close,
                        TradeReason(reason),
                    )
                    action = f"closed:{reason}"
                    detail = f"pnl=${closed.net_pnl:.4f}" if closed else ""
                else:
                    # Считаем прожитые свечи только при реальной смене бара.
                    if new_bar_closed:
                        open_pos.bars_held += 1
                    if action == "hold":
                        pass  # remain hold

            # 8. Вход: только если авто ВКЛ.
            #    С pyramiding вход возможен даже если есть открытые позиции
            #    (limit проверяется внутри portfolio engine).
            reject_reason = ""
            if self.is_auto(symbol, timeframe):
                # В live-режиме авто-торговля реальными деньгами запрещена,
                # пока settings.live_auto_enabled не включён (безопасность).
                if session.mode == SessionMode.live and not settings.live_auto_enabled:
                    logger.debug(
                        "LIVE авто %s %s: live_auto_enabled=false, пропуск",
                        symbol, timeframe,
                    )
                    action = "hold"
                else:
                    pred_return = pred.pred_return if pred else None
                    pred_min_low = pred.pred_min_low if pred else None
                    pred_max_high = pred.pred_max_high if pred else None

                    # Сначала пробуем LONG (прогноз на рост), затем SHORT (прогноз на падение).
                    # reject_reason фиксируем ТОЛЬКО для основного направления
                    # (которое показывается в гейдже), иначе причина SHORT'а
                    # затрёт причину LONG'а и введёт пользователя в заблуждение.
                    long_first = (pred_return is None) or (pred_return >= 0)
                    primary_reason = ""
                    if long_first:
                        entered = await self._try_enter_long(
                            portfolio, db, session, coin, snap, signal,
                            timeframe, pred_return, pred_max_high,
                        )
                        if not entered[0]:
                            primary_reason = entered[3]
                            entered = await self._try_enter_short(
                                portfolio, db, session, coin, snap, signal,
                                timeframe, pred_return, pred_min_low,
                            )
                    else:
                        entered = await self._try_enter_short(
                            portfolio, db, session, coin, snap, signal,
                            timeframe, pred_return, pred_min_low,
                        )
                        if not entered[0]:
                            primary_reason = entered[3]
                            entered = await self._try_enter_long(
                                portfolio, db, session, coin, snap, signal,
                                timeframe, pred_return, pred_max_high,
                            )

                    if entered[0]:  # (bool, action_str, detail_str, reject_reason)
                        action, detail = entered[1], entered[2]
                        reject_reason = ""
                    else:
                        reject_reason = primary_reason

            # 9. Commit транзакции (equity записывается ОДИН раз за весь тик
            #    в _tick_all_coins / refresh_all_predictions).
            await db.commit()

            # 9.1 Сохранить последний сигнал в кеш для live-отображения (без GPU).
            # win_prob — вероятность выбранного направления (одно число для UI).
            chosen_score = signal.s_entry if direction == "up" else signal.s_entry_short
            self._last_signals[f"{symbol}:{timeframe}"] = {
                "symbol": symbol,
                "tf": timeframe,
                "s_entry": signal.s_entry,
                "s_entry_short": signal.s_entry_short,
                "s_exit": signal.s_exit,
                "sub_signals": dict(signal.sub_signals),
                "sub_signals_short": dict(signal.sub_signals_short),
                "close": snap.close,
                "action": action,
                "direction": direction,
                "reject_reason": reject_reason,
                "pred_return": pred.pred_return if pred else None,
                "pred_path_slope": pred.pred_path_slope if pred else None,
                "win_prob": self.strategy.estimate_win_rate(
                    chosen_score, pred.pred_return if pred else None,
                ),
                "news_sentiment": news_result.sentiment if news_result else None,
                "news_summary": news_result.summary if news_result else None,
                "news_confidence": news_result.confidence if news_result else None,
                "news_sources": news_result.sources if news_result else [],
                "news_key_events": news_result.key_events if news_result else [],
                "news_fetched_at": (
                    news_result.fetched_at.isoformat() if news_result else None
                ),
                "updated_at": datetime.utcnow().isoformat(),
            }

            tick = TickResult(
                symbol=symbol, timeframe=timeframe,
                s_entry=signal.s_entry, s_exit=signal.s_exit,
                action=action, detail=detail,
                s_entry_short=signal.s_entry_short,
                direction=direction, close=snap.close,
            )

            # 10. Push в WebSocket (если подключён).
            if self.on_tick:
                try:
                    self.on_tick({
                        "symbol": symbol, "tf": timeframe,
                        "s_entry": signal.s_entry, "s_exit": signal.s_exit,
                        "s_entry_short": signal.s_entry_short,
                        "action": action, "close": snap.close,
                        "direction": direction,
                        "pred_return": pred.pred_return if pred else None,
                        "cash": session.cash,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                except Exception:
                    pass

            logger.info(
                "ТИК %s %s: S_long=%.3f S_short=%.3f dir=%s action=%s %s close=%.2f%s | subs=%s adx=%.1f rsi=%.1f",
                symbol, timeframe, signal.s_entry, signal.s_entry_short,
                direction, action, detail, snap.close,
                f" pred={pred.pred_return*100:+.2f}%" if pred else " pred=N/A",
                {k: round(v, 3) for k, v in signal.sub_signals.items()},
                snap.adx, snap.rsi,
            )
            return tick


# ─── Синглтон ────────────────────────────────────────────────

_loop: TradingLoop | None = None


def get_loop() -> TradingLoop:
    global _loop
    if _loop is None:
        _loop = TradingLoop()
    return _loop
