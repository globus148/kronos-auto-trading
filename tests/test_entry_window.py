"""Вход — только в первые минуты после закрытия свечи.

Баг: после перезапуска посреди 4h-бара бот заново проверял вход по свече,
которую уже оценил при её закрытии (память «какой бар проверен» живёт в
процессе), и открыл шорт INJ через 3.5 часа после закрытия свечи.
"""

from __future__ import annotations

import time

import pytest
from sqlalchemy import text

from config import settings
from core.trading_loop import ENTRY_WINDOW_SEC, TradingLoop
from db.database import engine, get_session, init_db
from db.models import Candle as CandleRow, Coin

TF_MS = 4 * 3600 * 1000


class _Market:
    async def is_active(self, symbol):
        return False  # ensure_history не ходит в сеть

    async def fetch_ticker(self, symbol):
        return {"last": 1.0, "bid": 0.0, "ask": 0.0, "volume": 0.0}

    async def fetch_ohlcv(self, *a, **k):
        return []


async def _setup(minutes_since_close: float):
    await init_db()
    async with engine.begin() as conn:
        for t in ["equity_point", "trade", "position", "session", "candle", "coin"]:
            await conn.execute(text(f"DELETE FROM {t}"))
    now = int(time.time() * 1000)
    last_closed_open = now - int(minutes_since_close * 60_000) - TF_MS
    n = settings.data_indicator_warmup + 60
    async with get_session() as db:
        coin = Coin(symbol="WIN/USDT", default_tf="4h")
        db.add(coin)
        await db.flush()
        rows = [{"coin_id": coin.id, "tf": "4h", "timestamp": last_closed_open - i * TF_MS,
                 "open": 1.0 + i * 1e-3, "high": 1.02 + i * 1e-3, "low": 0.98 + i * 1e-3,
                 "close": 1.0 + i * 1e-3, "volume": 1.0} for i in range(n)]
        # формирующаяся свеча
        rows.append({"coin_id": coin.id, "tf": "4h", "timestamp": last_closed_open + TF_MS,
                     "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1})
        await db.execute(CandleRow.__table__.insert(), rows)
        await db.commit()


def _patched_loop(monkeypatch):
    fake = _Market()
    monkeypatch.setattr("core.data_lifecycle.get_market", lambda: fake)
    monkeypatch.setattr("core.trading_loop.get_market", lambda: fake)
    monkeypatch.setattr(settings, "news_enabled", False)
    loop = TradingLoop()
    calls = []

    def fake_long(*a, **k):
        calls.append("long")
        return False, None, "тест"

    def fake_short(*a, **k):
        calls.append("short")
        return False, None, "тест"

    monkeypatch.setattr(loop.strategy, "evaluate_entry", fake_long)
    monkeypatch.setattr(loop.strategy, "evaluate_short_entry", fake_short)
    loop.set_auto("WIN/USDT", "4h", True)
    return loop, calls


@pytest.mark.asyncio
async def test_no_entry_long_after_bar_close(monkeypatch):
    await _setup(minutes_since_close=210)  # как с INJ: через 3.5 ч после закрытия
    loop, calls = _patched_loop(monkeypatch)
    tick = await loop.run_once("WIN/USDT", "4h")
    assert calls == []
    assert "после закрытия свечи" in loop._last_signals["WIN/USDT:4h"]["reject_reason"]
    assert tick.action == "hold"


@pytest.mark.asyncio
async def test_entry_checked_right_after_close(monkeypatch):
    await _setup(minutes_since_close=1)
    loop, calls = _patched_loop(monkeypatch)
    await loop.run_once("WIN/USDT", "4h")
    assert sorted(calls) == ["long", "short"]
    # повторный прогон по той же свече вход не проверяет
    await loop.run_once("WIN/USDT", "4h")
    assert len(calls) == 2


def test_window_is_short():
    assert ENTRY_WINDOW_SEC <= 30 * 60
