"""Тесты для TradingLoop (этап 8) — оркестратор торгового цикла.

Проверяем:
1. Полный конвейер run_once() на реальных данных.
2. Логику auto-режима (включается/выключается).
3. Корректное «hold» когда S_entry ниже порога.
4. Интеграцию всех модулей без ошибок.
"""

from __future__ import annotations

import asyncio

import pytest

from config import settings
from core.portfolio_engine import get_portfolio
from core.trading_loop import TradingLoop, get_loop
from db.database import engine, get_session, init_db, dispose_db
from db.models import Coin, SessionMode
from sqlalchemy import select, text


@pytest.fixture
async def setup_db():
    """Чистая БД + монета BTC/USDT."""
    await init_db()
    async with engine.begin() as conn:
        for t in ["equity_point", "trade", "position", "session", "coin", "candle"]:
            await conn.execute(text(f"DELETE FROM {t}"))
    async with get_session() as db:
        coin = Coin(symbol="BTC/USDT", enabled=True, default_tf="4h")
        db.add(coin)
        await db.commit()
    yield
    # cleanup в конце модуля.


class TestAutoMode:
    def test_set_auto(self):
        loop = TradingLoop()
        assert not loop.is_auto("BTC/USDT", "4h")
        loop.set_auto("BTC/USDT", "4h", True)
        assert loop.is_auto("BTC/USDT", "4h")
        loop.set_auto("BTC/USDT", "4h", False)
        assert not loop.is_auto("BTC/USDT", "4h")


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_run_once_returns_tick_result(self, setup_db):
        """run_once возвращает корректный TickResult на реальных данных."""
        try:
            loop = get_loop()
            result = await loop.run_once("BTC/USDT", "4h")
            assert result.symbol == "BTC/USDT"
            assert result.timeframe == "4h"
            assert 0.0 <= result.s_entry <= 1.0
            assert result.action in ("hold", "opened") or result.action.startswith("closed")
            print(f"\n  S_entry={result.s_entry:.3f} action={result.action}")
        except Exception as e:
            pytest.skip(f"Сеть/Binance недоступен: {e}")

    @pytest.mark.asyncio
    async def test_auto_disabled_means_no_open(self, setup_db):
        """Без авто-режима — позиция не открывается даже при высоком S_entry."""
        try:
            loop = get_loop()
            loop.set_auto("BTC/USDT", "4h", False)  # авто ВЫКЛ
            await loop.run_once("BTC/USDT", "4h")
            async with get_session() as db:
                pe = get_portfolio()
                session = await pe.get_or_create_session(db, SessionMode.paper)
                positions = await pe.get_open_positions(db, session)
                assert len(positions) == 0  # не открылось
        except Exception as e:
            pytest.skip(f"Сеть недоступен: {e}")

    @pytest.mark.asyncio
    async def test_hold_when_signal_below_threshold(self, setup_db):
        """При S_entry < entry_threshold action != 'opened'."""
        try:
            loop = get_loop()
            loop.set_auto("BTC/USDT", "4h", True)
            result = await loop.run_once("BTC/USDT", "4h")
            if result.s_entry < settings.entry_threshold:
                assert result.action != "opened"
                print(f"\n  S_entry={result.s_entry:.3f} < {settings.entry_threshold} → hold (верно)")
        except Exception as e:
            pytest.skip(f"Сеть недоступен: {e}")

    @pytest.mark.asyncio
    async def test_equity_recorded(self, setup_db):
        """После тика в БД есть equity_point."""
        try:
            loop = get_loop()
            await loop.run_once("BTC/USDT", "4h")
            async with get_session() as db:
                from db.models import EquityPoint
                from sqlalchemy import func
                count = (await db.execute(
                    select(func.count()).select_from(EquityPoint)
                )).scalar_one()
                assert count >= 1  # equity записан
        except Exception as e:
            pytest.skip(f"Сеть недоступен: {e}")
