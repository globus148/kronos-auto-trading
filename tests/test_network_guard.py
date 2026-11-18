"""Тесты: пауза торговли при обрыве связи и чистые логи.

1. NetworkGuard: обрыв → одна строка WARNING, возврат связи → одна строка INFO.
2. Цикл: первая сетевая ошибка останавливает прогон (остальные монеты не
   опрашиваются), без traceback'ов; пока связи нет — прогоны пропускаются.
3. Фильтр убирает дубли «gaierror exception in shielded future» из лога asyncio.
4. Размер позиции считается от капитала (cash + позиции), а не от остатка cash.
"""

from __future__ import annotations

import logging
import socket

import ccxt
import pytest
from sqlalchemy import text

import core.trading_loop as tl
from core.network_guard import DropNetworkNoise, NetworkGuard, is_network_error
from core.trading_loop import TickResult, TradingLoop
from db.database import engine, get_session, init_db
from db.models import Coin


def _offline_exc() -> Exception:
    exc = ccxt.RequestTimeout("binance GET https://api.binance.com/api/v3/klines?...")
    exc.__cause__ = socket.gaierror(11001, "getaddrinfo failed")
    return exc


class TestNetworkGuard:
    @pytest.mark.asyncio
    async def test_offline_then_back_online_logs_two_lines(self, caplog):
        state = {"up": False}

        async def probe():
            if not state["up"]:
                raise _offline_exc()

        guard = NetworkGuard(probe=probe)
        caplog.set_level(logging.INFO, logger="trading.network")

        guard.report_failure(_offline_exc())
        guard.report_failure(_offline_exc())  # повтор не пишет вторую строку
        assert guard.online is False
        assert "DNS" in guard.status()["reason"]

        assert await guard.ensure_online() is False
        assert await guard.check() is False

        state["up"] = True
        assert await guard.check() is True
        assert guard.online is True
        assert await guard.check() is False  # уже online — ничего не делает

        msgs = [r.getMessage() for r in caplog.records if r.name == "trading.network"]
        assert len(msgs) == 2
        assert msgs[0].startswith("НЕТ СВЯЗИ")
        assert msgs[1].startswith("Связь с биржей восстановлена")

    def test_network_error_classification(self):
        assert is_network_error(_offline_exc())
        assert is_network_error(TimeoutError())
        assert not is_network_error(ValueError("bug"))
        assert not is_network_error(ccxt.InsufficientFunds("x"))

    def test_asyncio_noise_filter(self):
        f = DropNetworkNoise()
        try:
            raise socket.gaierror(11001, "getaddrinfo failed")
        except socket.gaierror:
            import sys
            rec = logging.LogRecord("asyncio", logging.ERROR, __file__, 1,
                                    "gaierror exception in shielded future", None, sys.exc_info())
        assert f.filter(rec) is False
        other = logging.LogRecord("asyncio", logging.ERROR, __file__, 1, "Task was destroyed", None, None)
        assert f.filter(other) is True


@pytest.fixture
async def coins_db():
    await init_db()
    async with engine.begin() as conn:
        for t in ["equity_point", "trade", "position", "session", "candle", "coin"]:
            await conn.execute(text(f"DELETE FROM {t}"))
    async with get_session() as db:
        db.add_all([Coin(symbol=f"C{i}/USDT", enabled=True) for i in range(5)])
        await db.commit()
    yield


class TestLoopPausesWhenOffline:
    @pytest.mark.asyncio
    async def test_first_network_error_stops_the_pass(self, coins_db, monkeypatch, caplog):
        state = {"up": False}

        async def probe():
            if not state["up"]:
                raise _offline_exc()

        guard = NetworkGuard(probe=probe)
        monkeypatch.setattr(tl, "get_network_guard", lambda: guard)
        calls = []

        async def fake_run_once(self, symbol, timeframe):
            calls.append(symbol)
            if not state["up"]:
                raise _offline_exc()
            return TickResult(symbol, timeframe, 0, 0, "hold", "", close=1.0)

        monkeypatch.setattr(TradingLoop, "run_once", fake_run_once)
        loop = TradingLoop()
        caplog.set_level(logging.INFO)

        await loop._tick_all_coins("4h")
        assert len(calls) == 1          # остальные 4 монеты не опрашивались
        assert guard.online is False
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

        await loop._tick_all_coins("4h")
        await loop.refresh_all_predictions()
        assert len(calls) == 1          # пока связи нет — ничего не делаем

        state["up"] = True
        await loop._tick_all_coins("4h")
        assert len(calls) == 6          # связь вернулась → все 5 монет
        assert guard.online is True

    @pytest.mark.asyncio
    async def test_health_reports_network(self):
        assert "network" in TradingLoop().get_health()


class TestSizingUsesEquity:
    @pytest.mark.asyncio
    async def test_equity_includes_open_positions(self, coins_db):
        from core.portfolio_engine import PortfolioEngine
        from core.risk_manager import RiskManager

        pe = PortfolioEngine(commission_rate=0.001, slippage_rate=0.0005)
        rm = RiskManager(risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0,
                         max_position_pct=0.30, commission_rate=0.001, slippage_rate=0.0005)
        async with get_session() as db:
            coin = Coin(symbol="EQ/USDT", enabled=True)
            db.add(coin)
            await db.flush()
            session = await pe.get_or_create_session(db)
            sizing = rm.calculate(entry_price=100.0, atr=2.0, equity=session.cash, cash=session.cash)
            assert await pe.open_position(db, session, coin, sizing, "4h") is not None
            await db.commit()
            equity = await TradingLoop()._equity(db, session)
            assert session.cash < 80
            assert equity == pytest.approx(session.cash + sizing.qty * 100.0 * (1 + 0.0005), rel=0.01)
            assert equity > 99.0
