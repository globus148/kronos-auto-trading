"""Unit-тесты для ручного открытия позиций (open_manual_position).

Проверяем:
1. Ручной LONG: создаёт позицию и trade с заданными qty/price.
2. Ручной SHORT: создаёт short-позицию.
3. Стопы считаются из ATR (если передан).
4. Без ATR стопы = 0 (только ручное закрытие).
5. Некорректные qty/price отклоняются.
6. Недостаточно cash — пропуск.
7. Short в live отклоняется.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from config import settings
from core.portfolio_engine import PortfolioEngine
from db.database import engine, get_session, init_db
from db.models import Coin, SessionMode, TradeReason


# ─── Фикстуры ───────────────────────────────────────────────


@pytest.fixture
async def clean_db():
    await init_db()
    async with engine.begin() as conn:
        for t in ["equity_point", "trade", "position", "session", "coin", "strategy_config"]:
            await conn.execute(text(f"DELETE FROM {t}"))
    yield


@pytest.fixture
def portfolio() -> PortfolioEngine:
    return PortfolioEngine()


@pytest.fixture
async def btc(clean_db):
    async with get_session() as db:
        coin = Coin(symbol="BTC/USDT", enabled=True, default_tf="4h")
        db.add(coin)
        await db.commit()
        await db.refresh(coin)
        return coin


# ─── Тесты ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manual_long_with_atr(clean_db, btc):
    """Ручной LONG с ATR: стопы рассчитываются автоматически."""
    portfolio = PortfolioEngine()
    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.paper)
        result = await portfolio.open_manual_position(
            db, session, btc, side="long", qty=0.001, entry_price=65000.0,
            timeframe="4h", atr=1500.0,
        )
        assert result is not None
        position, execution = result
        assert position.qty == 0.001
        assert position.entry_price == 65000.0
        assert position.side.value == "long"
        # Стоп = entry - 1.5 * ATR = 65000 - 2250 = 62750
        assert position.stop_price == 65000.0 - 1.5 * 1500.0
        # Тейк = entry + 3.0 * ATR = 65000 + 4500 = 69500
        assert position.target_price == 65000.0 + 3.0 * 1500.0
        # Комиссия списана.
        assert execution.fee > 0


@pytest.mark.asyncio
async def test_manual_short_with_atr(clean_db, btc):
    """Ручной SHORT с ATR: стопы выше входа, тейк ниже."""
    portfolio = PortfolioEngine()
    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.paper)
        result = await portfolio.open_manual_position(
            db, session, btc, side="short", qty=0.001, entry_price=65000.0,
            timeframe="4h", atr=1500.0,
        )
        assert result is not None
        position, _ = result
        assert position.side.value == "short"
        # Стоп short = entry + 1.5 * ATR = 67250
        assert position.stop_price == 65000.0 + 1.5 * 1500.0
        # Тейк short = entry - 3.0 * ATR = 60500
        assert position.target_price == 65000.0 - 3.0 * 1500.0


@pytest.mark.asyncio
async def test_manual_no_atr_no_stops(clean_db, btc):
    """Без ATR стопы = 0 (только ручное закрытие)."""
    portfolio = PortfolioEngine()
    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.paper)
        result = await portfolio.open_manual_position(
            db, session, btc, side="long", qty=0.001, entry_price=65000.0,
            timeframe="4h", atr=None,
        )
        assert result is not None
        position, _ = result
        assert position.stop_price == 0.0
        assert position.target_price == 0.0


@pytest.mark.asyncio
async def test_manual_invalid_params(clean_db, btc):
    """Некорректные qty/price → None."""
    portfolio = PortfolioEngine()
    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.paper)
        # qty = 0
        assert await portfolio.open_manual_position(
            db, session, btc, "long", 0, 65000, "4h",
        ) is None
        # price = 0
        assert await portfolio.open_manual_position(
            db, session, btc, "long", 0.001, 0, "4h",
        ) is None


@pytest.mark.asyncio
async def test_manual_insufficient_cash(clean_db, btc):
    """Недостаточно cash → None (сделка не открывается)."""
    portfolio = PortfolioEngine()
    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.paper)
        # Пытаемся купить на $1_000_000 при балансе $100.
        result = await portfolio.open_manual_position(
            db, session, btc, "long", qty=10.0, entry_price=100000.0,
            timeframe="4h",
        )
        assert result is None
        # Cash не изменился.
        assert session.cash == settings.paper_initial_balance


@pytest.mark.asyncio
async def test_manual_short_rejected_in_live(clean_db, btc):
    """Short в live-режиме отклоняется (spot не поддерживает)."""
    portfolio = PortfolioEngine()
    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.live)
        result = await portfolio.open_manual_position(
            db, session, btc, "short", qty=0.001, entry_price=65000.0,
            timeframe="4h",
        )
        assert result is None


@pytest.mark.asyncio
async def test_manual_long_deducts_cash_with_fee(clean_db, btc):
    """Cash уменьшается на position_value + fee."""
    portfolio = PortfolioEngine()
    qty = 0.001
    price = 65000.0
    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.paper)
        cash_before = session.cash
        result = await portfolio.open_manual_position(
            db, session, btc, "long", qty=qty, entry_price=price,
            timeframe="4h",
        )
        await db.commit()
        assert result is not None
        session = await portfolio.get_or_create_session(db, SessionMode.paper)
        # Cash должен уменьшиться на qty*price + fee.
        fee = qty * price * portfolio.commission_rate
        expected_cash = cash_before - qty * price - fee
        assert abs(session.cash - expected_cash) < 0.0001
