"""Unit-тесты для live-ветки PortfolioEngine (реальные ордера, mock биржи).

Проверяем:
1. open_position в live-режиме ставит реальный ордер через LiveBroker.
2. close_position в live-режиме ставит реальный ордер.
3. cash синхронизируется с балансом биржи после операций.
4. Шорт в live-режиме отклоняется (spot не поддерживает).
5. Если нет кошелька — live-операция пропускается (None).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import delete, text

from config import settings
from core.portfolio_engine import PortfolioEngine
from core.risk_manager import PositionSizing
from db.database import engine, get_session, init_db
from db.models import (
    Coin,
    PositionStatus,
    Session,
    SessionMode,
    TradeReason,
)


# ─── Фикстуры ───────────────────────────────────────────────


@pytest.fixture
async def clean_db():
    """Чистая БД для каждого теста."""
    await init_db()
    async with engine.begin() as conn:
        for t in [
            "equity_point", "trade", "position", "session",
            "coin", "strategy_config",
        ]:
            await conn.execute(text(f"DELETE FROM {t}"))
    yield


@pytest.fixture
def portfolio() -> PortfolioEngine:
    return PortfolioEngine()


@pytest.fixture
async def btc(clean_db):
    """Монета BTC/USDT."""
    async with get_session() as db:
        coin = Coin(symbol="BTC/USDT", enabled=True, default_tf="4h")
        db.add(coin)
        await db.commit()
        await db.refresh(coin)
        return coin


def make_sizing(qty: float = 0.001, value: float = 10.0) -> PositionSizing:
    """Фабрика PositionSizing для тестов."""
    return PositionSizing(
        qty=qty,
        position_value=value,
        stop_price=50000.0,
        target_price=52000.0,
        risk_amount=0.5,
        risk_pct=0.01,
        stop_pct=0.03,
        rr_ratio=2.0,
        break_even_price=50050.0,
    )


def mock_order_result(qty: float = 0.001, avg_price: float = 65000.0,
                      fee: float = 0.005) -> "MockOrderResult":
    """Mock LiveOrderResult для подстановки вместо биржи."""
    from core.market_service import LiveOrderResult
    return LiveOrderResult(
        symbol="BTC/USDT", side="buy",
        qty=qty, avg_price=avg_price,
        cost=qty * avg_price, fee=fee,
        order_id="mock_order_123", raw={},
    )


# ─── Тесты ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_position_live_puts_real_order(clean_db, btc):
    """Live open_position вызывает LiveBroker.create_market_buy."""
    portfolio = PortfolioEngine()
    sizing = make_sizing(qty=0.001, value=65.0)

    order = mock_order_result(qty=0.001, avg_price=65000.0, fee=0.006)
    mock_broker = AsyncMock()
    mock_broker.create_market_buy = AsyncMock(return_value=order)
    mock_broker.fetch_usdt_balance = AsyncMock(return_value=35.0)
    mock_broker.close = AsyncMock()

    with patch.object(
        portfolio, "_get_live_broker", new_callable=AsyncMock,
        return_value=mock_broker,
    ):
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.live)
            result = await portfolio.open_position(
                db, session, btc, sizing, "4h", TradeReason.manual,
            )

    # Ордер был вызван с правильными параметрами.
    mock_broker.create_market_buy.assert_called_once_with("BTC/USDT", 65.0)
    # Sync cash после ордера.
    mock_broker.fetch_usdt_balance.assert_called()

    assert result is not None
    position, execution = result
    assert position.qty == 0.001
    assert position.entry_price == 65000.0
    assert execution.fee == 0.006


@pytest.mark.asyncio
async def test_close_position_live_puts_real_sell(clean_db, btc):
    """Live close_position вызывает LiveBroker.create_market_sell."""
    portfolio = PortfolioEngine()
    sizing = make_sizing(qty=0.002, value=130.0)

    # Сначала открываем live-позицию.
    order_buy = mock_order_result(qty=0.002, avg_price=65000.0, fee=0.012)
    order_sell = mock_order_result(qty=0.002, avg_price=67000.0, fee=0.013)
    # override side для sell
    order_sell = type(order_sell)(
        symbol="BTC/USDT", side="sell",
        qty=0.002, avg_price=67000.0,
        cost=0.002 * 67000.0, fee=0.013,
        order_id="mock_sell_456", raw={},
    )

    # Mock broker для open.
    mock_broker_open = AsyncMock()
    mock_broker_open.create_market_buy = AsyncMock(return_value=order_buy)
    mock_broker_open.fetch_usdt_balance = AsyncMock(return_value=870.0)
    mock_broker_open.close = AsyncMock()

    # Mock broker для close.
    mock_broker_close = AsyncMock()
    mock_broker_close.create_market_sell = AsyncMock(return_value=order_sell)
    mock_broker_close.fetch_usdt_balance = AsyncMock(return_value=1134.0)
    mock_broker_close.close = AsyncMock()

    with patch.object(
        portfolio, "_get_live_broker", new_callable=AsyncMock,
        return_value=mock_broker_open,
    ):
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.live)
            open_res = await portfolio.open_position(
                db, session, btc, sizing, "4h", TradeReason.manual,
            )
            await db.commit()

    assert open_res is not None
    position = open_res[0]

    # Теперь закрываем.
    with patch.object(
        portfolio, "_get_live_broker", new_callable=AsyncMock,
        return_value=mock_broker_close,
    ):
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.live)
            pos_db = await portfolio.get_open_position(db, session, btc)
            closed = await portfolio.close_position(
                db, session, pos_db, btc, 67000.0, TradeReason.manual,
            )

    mock_broker_close.create_market_sell.assert_called_once_with("BTC/USDT", 0.002)
    assert closed is not None
    assert closed.net_pnl != 0  # $4 прибыль (67000 - 65000) * 0.002


@pytest.mark.asyncio
async def test_open_short_rejected_in_live(clean_db, btc):
    """Short в live spot отклоняется (недоступен на spot бирже)."""
    portfolio = PortfolioEngine()
    sizing = make_sizing(qty=0.001, value=10.0)

    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.live)
        result = await portfolio.open_short_position(
            db, session, btc, sizing, "4h", TradeReason.manual,
        )

    assert result is None


@pytest.mark.asyncio
async def test_live_open_no_wallet_returns_none(clean_db, btc):
    """Если кошелька нет — live open_position возвращает None."""
    portfolio = PortfolioEngine()
    sizing = make_sizing()

    with patch.object(
        portfolio, "_get_live_broker", new_callable=AsyncMock, return_value=None,
    ):
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.live)
            result = await portfolio.open_position(
                db, session, btc, sizing, "4h", TradeReason.manual,
            )

    assert result is None


@pytest.mark.asyncio
async def test_live_session_starts_with_zero_balance(clean_db):
    """Live-сессия создаётся с initial_balance=0 (cash подгрузится с биржи)."""
    portfolio = PortfolioEngine()
    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.live)
        assert session.mode == SessionMode.live
        assert session.initial_balance == 0.0
        assert session.cash == 0.0


@pytest.mark.asyncio
async def test_paper_session_unchanged(clean_db, btc):
    """Paper-режим работает как раньше (без изменений)."""
    portfolio = PortfolioEngine()
    sizing = make_sizing(qty=0.001, value=10.0)

    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, SessionMode.paper)
        assert session.mode == SessionMode.paper
        assert session.initial_balance == settings.paper_initial_balance
        # Paper-метод не трогает биржу.
        result = await portfolio.open_position(
            db, session, btc, sizing, "4h", TradeReason.manual,
        )

    assert result is not None
    position = result[0]
    assert position.qty == 0.001
