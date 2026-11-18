"""Тесты: закрытие позиций при остановке, единое время (UTC), устаревшие данные.

1. TradingLoop.close_open_positions закрывает все открытые позиции (long и
   short) с причиной shutdown и возвращает деньги в cash.
2. stop() запущенного цикла закрывает позиции; после остановки run_once
   ничего не делает.
3. Логи и планировщик работают в UTC.
4. DataLifecycle не качает и не трогает данные пары, остановленной на бирже,
   и не удаляет историю, если биржа не отдала ничего новее.
5. run_once пропускает пару с устаревшими свечами.
"""

from __future__ import annotations

import logging
import time

import pytest
from sqlalchemy import select, text

from core.portfolio_engine import PortfolioEngine
from core.risk_manager import RiskManager
from core.trading_loop import TradingLoop
from db.database import engine, get_session, init_db
from db.models import (
    Candle as CandleRow,
    Coin,
    Position,
    PositionStatus,
    Trade,
    TradeReason,
)


@pytest.fixture
async def clean_db():
    await init_db()
    async with engine.begin() as conn:
        for t in ["equity_point", "trade", "position", "session", "candle", "coin"]:
            await conn.execute(text(f"DELETE FROM {t}"))
    yield


async def _open_long_and_short() -> float:
    """Открыть 1 long (AAA) и 1 short (BBB) в paper-сессии. Возвращает cash после."""
    pe = PortfolioEngine(commission_rate=0.001, slippage_rate=0.0005)
    rm = RiskManager(
        risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0,
        max_position_pct=0.30, commission_rate=0.001, slippage_rate=0.0005,
    )
    async with get_session() as db:
        a = Coin(symbol="AAA/USDT", default_tf="1h")
        b = Coin(symbol="BBB/USDT", default_tf="4h")
        db.add_all([a, b])
        await db.flush()
        session = await pe.get_or_create_session(db)
        long_sizing = rm.calculate(
            entry_price=100.0, atr=2.0, equity=session.cash, cash=session.cash,
        )
        assert await pe.open_position(db, session, a, long_sizing, "1h") is not None
        short_sizing = rm.calculate_short(
            entry_price=50.0, atr=1.0, equity=session.cash, cash=session.cash,
        )
        assert await pe.open_short_position(db, session, b, short_sizing, "4h") is not None
        await db.commit()
        return session.cash


class _FakeMarket:
    """Заглушка биржи: тикер по словарю, без сети."""

    def __init__(self, prices: dict[str, float], active: dict[str, bool] | None = None):
        self.prices = prices
        self.active = active or {}
        self.history_calls = 0
        self.history: list = []

    async def fetch_ticker(self, symbol: str) -> dict:
        if symbol not in self.prices:
            raise RuntimeError("нет сети")
        return {"last": self.prices[symbol], "bid": 0.0, "ask": 0.0, "volume": 0.0}

    async def is_active(self, symbol: str):
        return self.active.get(symbol)

    async def fetch_ohlcv_history(self, symbol, timeframe, total, end_ts_ms=None):
        self.history_calls += 1
        return list(self.history)

    async def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=500):
        return []


class TestCloseOnShutdown:
    @pytest.mark.asyncio
    async def test_close_open_positions_closes_everything(self, clean_db, monkeypatch):
        cash_after_open = await _open_long_and_short()
        fake = _FakeMarket({"AAA/USDT": 101.0, "BBB/USDT": 49.0})
        monkeypatch.setattr("core.trading_loop.get_market", lambda: fake)

        loop = TradingLoop()
        closed = await loop.close_open_positions("тест")
        assert closed == 2

        async with get_session() as db:
            positions = (await db.execute(select(Position))).scalars().all()
            assert len(positions) == 2
            assert all(p.status == PositionStatus.closed for p in positions)
            exits = (await db.execute(
                select(Trade).where(Trade.reason == TradeReason.shutdown)
            )).scalars().all()
            assert len(exits) == 2
            pe = PortfolioEngine()
            session = await pe.get_or_create_session(db)
            # Деньги вернулись из позиций, обе сделки в плюс (цена пошла в нашу сторону).
            assert session.cash > cash_after_open
            assert sum(t.pnl for t in exits) > 0
            assert session.cash > 100.0

        # Повторный вызов — закрывать уже нечего.
        assert await loop.close_open_positions("тест") == 0

    @pytest.mark.asyncio
    async def test_price_falls_back_to_last_candle(self, clean_db, monkeypatch):
        """Тикер недоступен → берём close последней свечи из БД."""
        await _open_long_and_short()
        async with get_session() as db:
            coins = {c.symbol: c for c in (await db.execute(select(Coin))).scalars().all()}
            now_ms = int(time.time() * 1000)
            db.add(CandleRow(coin_id=coins["AAA/USDT"].id, tf="1h", timestamp=now_ms - 3_600_000,
                             open=100, high=103, low=99, close=102.0, volume=1))
            await db.commit()
        fake = _FakeMarket({"BBB/USDT": 50.0})  # по AAA тикера нет
        monkeypatch.setattr("core.trading_loop.get_market", lambda: fake)

        assert await TradingLoop().close_open_positions("тест") == 2
        async with get_session() as db:
            exit_long = (await db.execute(
                select(Trade).join(Position).join(Coin, Coin.id == Position.coin_id)
                .where(Coin.symbol == "AAA/USDT", Trade.reason == TradeReason.shutdown)
            )).scalars().one()
            assert exit_long.price == pytest.approx(102.0 * (1 - 0.0005))

    @pytest.mark.asyncio
    async def test_stop_closes_positions_and_blocks_ticks(self, clean_db, monkeypatch):
        await _open_long_and_short()
        fake = _FakeMarket({"AAA/USDT": 100.0, "BBB/USDT": 50.0})
        monkeypatch.setattr("core.trading_loop.get_market", lambda: fake)

        loop = TradingLoop()
        await loop.start()
        await loop.stop()

        async with get_session() as db:
            open_left = (await db.execute(
                select(Position).where(Position.status == PositionStatus.open)
            )).scalars().all()
            assert open_left == []
        tick = await loop.run_once("AAA/USDT", "1h")
        assert tick.action == "skip"


class TestUtcEverywhere:
    def test_scheduler_and_cron_in_utc(self):
        from apscheduler.triggers.cron import CronTrigger
        from core.timeutil import SCHEDULER_TZ

        loop = TradingLoop()
        assert str(loop.scheduler.timezone) == "UTC"
        assert str(CronTrigger(minute=1, timezone=SCHEDULER_TZ).timezone) == "UTC"

    @pytest.mark.asyncio
    async def test_jobs_scheduled_in_utc(self, clean_db, monkeypatch):
        monkeypatch.setattr("core.trading_loop.get_market", lambda: _FakeMarket({}))
        loop = TradingLoop()
        await loop.start()
        try:
            jobs = loop.scheduler.get_jobs()
            assert jobs
            for job in jobs:
                assert str(job.trigger.timezone) == "UTC"
                assert job.next_run_time.utcoffset().total_seconds() == 0
            tick_4h = loop.scheduler.get_job("tick_4h")
            assert tick_4h.next_run_time.hour % 4 == 0  # 0,4,8… UTC = закрытие 4h-свечи
        finally:
            loop.scheduler.shutdown(wait=False)

    def test_log_formatters_use_utc(self, tmp_path, monkeypatch):
        """Файловые логи пишут время в UTC (проверяем на временной папке)."""
        import core.logging_service as ls
        from logging.handlers import RotatingFileHandler

        monkeypatch.setattr(ls, "LOG_DIR", tmp_path)
        monkeypatch.setattr(ls, "TRADING_LOG_PATH", tmp_path / "trading.log")
        monkeypatch.setattr(ls, "TRADES_LOG_PATH", tmp_path / "trades.log")
        monkeypatch.setattr(ls, "SIGNALS_CSV_PATH", tmp_path / "signals.csv")
        root, trades = logging.getLogger(), logging.getLogger("trades")
        before = (list(root.handlers), list(trades.handlers), root.level)
        try:
            ls.setup_logging()
            added = [h for h in root.handlers + trades.handlers
                     if h not in before[0] + before[1]]
            files = [h for h in added if isinstance(h, RotatingFileHandler)]
            assert len(files) == 2
            for h in files:
                assert h.formatter.converter is time.gmtime
            ls.log_trade_reject("AAA/USDT", "1h", "long", 0.1, "тест")
            for h in files:
                h.flush()
            line = (tmp_path / "trades.log").read_text(encoding="utf-8").splitlines()[-1]
            assert line.startswith(time.strftime("%Y-%m-%d %H:", time.gmtime()))
        finally:
            for h in root.handlers[:]:
                if h not in before[0]:
                    root.removeHandler(h); h.close()
            for h in trades.handlers[:]:
                if h not in before[1]:
                    trades.removeHandler(h); h.close()
            root.setLevel(before[2])

    def test_time_helpers(self):
        from datetime import datetime, timedelta, timezone
        from core.timeutil import ms_to_utc_str, utc_str, utcnow

        assert utcnow().tzinfo is None
        assert ms_to_utc_str(0) == "1970-01-01 00:00:00"
        msk = datetime(2026, 11, 11, 20, 1, 5, tzinfo=timezone(timedelta(hours=3)))
        assert utc_str(msk) == "2026-11-11 17:01:05"


class TestInactiveAndStalePairs:
    @pytest.mark.asyncio
    async def test_inactive_pair_is_not_fetched_or_wiped(self, clean_db, monkeypatch):
        from core.data_lifecycle import DataLifecycleManager

        async with get_session() as db:
            coin = Coin(symbol="DEAD/USDT", default_tf="4h")
            db.add(coin)
            await db.flush()
            old = int(time.time() * 1000) - 200 * 86_400_000
            db.add_all([
                CandleRow(coin_id=coin.id, tf="4h", timestamp=old + i * 14_400_000,
                          open=1, high=1, low=1, close=1.6, volume=1)
                for i in range(5)
            ])
            await db.commit()

            fake = _FakeMarket({}, active={"DEAD/USDT": False})
            monkeypatch.setattr("core.data_lifecycle.get_market", lambda: fake)
            report = await DataLifecycleManager().ensure_history(db, coin, "4h")

            assert report.action == "inactive"
            assert fake.history_calls == 0
            assert report.final_count == 5

    @pytest.mark.asyncio
    async def test_full_refresh_keeps_data_when_exchange_has_nothing_newer(
        self, clean_db, monkeypatch,
    ):
        from config import settings
        from core.data_lifecycle import DataLifecycleManager
        from core.market_service import Candle

        n = int(settings.required_history * 0.6)  # ≥50% (не cold start), но < required
        tf_ms = 14_400_000
        # Свежее retention-окна (не попадают под pruning), но отстают на 100 баров.
        last = (int(time.time() * 1000) // tf_ms - 100) * tf_ms
        async with get_session() as db:
            coin = Coin(symbol="OLD/USDT", default_tf="4h")
            db.add(coin)
            await db.flush()
            await db.execute(CandleRow.__table__.insert(), [
                {"coin_id": coin.id, "tf": "4h", "timestamp": last - i * tf_ms,
                 "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
                for i in range(n)
            ])
            await db.commit()

            fake = _FakeMarket({}, active={"OLD/USDT": True})
            fake.history = [Candle(last, 1, 1, 1, 1, 1)]  # ничего новее, чем в БД
            monkeypatch.setattr("core.data_lifecycle.get_market", lambda: fake)
            report = await DataLifecycleManager().ensure_history(db, coin, "4h")

            assert report.action == "stale"
            assert report.final_count == n  # история не удалена

    @pytest.mark.asyncio
    async def test_run_once_skips_stale_pair(self, clean_db, monkeypatch):
        from config import settings

        tf_ms = 14_400_000
        n = settings.data_indicator_warmup + 50
        old_last = int(time.time() * 1000) - 100 * tf_ms
        async with get_session() as db:
            coin = Coin(symbol="DEAD/USDT", default_tf="4h")
            db.add(coin)
            await db.flush()
            await db.execute(CandleRow.__table__.insert(), [
                {"coin_id": coin.id, "tf": "4h", "timestamp": old_last - i * tf_ms,
                 "open": 1.6, "high": 1.6, "low": 1.6, "close": 1.6, "volume": 1.0}
                for i in range(n)
            ])
            await db.commit()

        fake = _FakeMarket({}, active={"DEAD/USDT": False})
        monkeypatch.setattr("core.data_lifecycle.get_market", lambda: fake)
        monkeypatch.setattr("core.trading_loop.get_market", lambda: fake)

        loop = TradingLoop()
        loop.set_auto("DEAD/USDT", "4h", True)
        tick = await loop.run_once("DEAD/USDT", "4h")

        assert tick.action == "skip"
        assert "устарели" in tick.detail
        async with get_session() as db:
            assert (await db.execute(select(Position))).scalars().all() == []

    @pytest.mark.asyncio
    async def test_halted_pair_is_disabled_and_restored(self, clean_db, monkeypatch):
        """Остановленная на бирже пара выключается (уходит с дашборда) и
        включается обратно, когда биржа возобновляет торги."""
        from config import settings

        symbol = settings.default_coins[0]
        async with get_session() as db:
            db.add_all([Coin(symbol=symbol, enabled=True), Coin(symbol="XXX/USDT", enabled=True)])
            await db.commit()

        fake = _FakeMarket({}, active={symbol: False, "XXX/USDT": True})
        monkeypatch.setattr("core.trading_loop.get_market", lambda: fake)
        loop = TradingLoop()
        await loop.sync_coin_availability()
        async with get_session() as db:
            enabled = {c.symbol for c in (await db.execute(
                select(Coin).where(Coin.enabled == True))).scalars().all()}
        assert enabled == {"XXX/USDT"}

        fake.active[symbol] = True
        await loop.sync_coin_availability()
        async with get_session() as db:
            enabled = {c.symbol for c in (await db.execute(
                select(Coin).where(Coin.enabled == True))).scalars().all()}
        assert enabled == {symbol, "XXX/USDT"}

        # Статус неизвестен (нет сети) — ничего не меняем.
        fake.active.clear()
        await loop.sync_coin_availability()
        async with get_session() as db:
            assert len((await db.execute(
                select(Coin).where(Coin.enabled == True))).scalars().all()) == 2
