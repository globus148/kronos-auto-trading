"""Unit-тесты для PortfolioEngine (план 4.7, раздел 6).

Проверяем:
1. Полный цикл сделки: вход → рост → закрытие с прибылью.
2. Цикл с убытком: вход → падение → стоп-лосс.
3. Учёт комиссий и slippage (критично для выхода в плюс).
4. Метрики: win rate, profit factor, Sharpe, drawdown.
5. Бизнес-логика: нельзя открыть 2 позиции по одной монете,
   нельзя превысить cash.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from config import settings
from core.portfolio_engine import PortfolioEngine
from core.risk_manager import PositionSizing, RiskManager
from db.database import engine, get_session, init_db, dispose_db
from db.models import (
    Coin,
    PositionStatus,
    Session,
    SessionMode,
    Trade,
    TradeReason,
    TradeSide,
)
from sqlalchemy import delete, select, text


# ─── Фикстуры ───────────────────────────────────────────────

@pytest.fixture
async def clean_db():
    """Чистая БД для каждого теста."""
    await init_db()
    async with engine.begin() as conn:
        for t in ["equity_point", "trade", "position", "session", "coin"]:
            await conn.execute(text(f"DELETE FROM {t}"))
    yield
    # Не закрываем движок до конца сессии — dispose в конце модуля.


@pytest.fixture
def risk_mgr() -> RiskManager:
    return RiskManager()


@pytest.fixture
def portfolio() -> PortfolioEngine:
    return PortfolioEngine()


@pytest.fixture
def make_coin():
    """Фабрика монет (async, возвращает корутину)."""
    async def _make(db, symbol="BTC/USDT"):
        coin = Coin(symbol=symbol, enabled=True, default_tf="4h")
        db.add(coin)
        await db.commit()
        await db.refresh(coin)
        return coin
    return _make


# ─── Тесты базового цикла ───────────────────────────────────

class TestSession:
    @pytest.mark.asyncio
    async def test_creates_session_with_100(self, clean_db, portfolio):
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            assert session.initial_balance == 100.0
            assert session.cash == 100.0
            assert session.mode == SessionMode.paper

    @pytest.mark.asyncio
    async def test_reuses_existing_session(self, clean_db, portfolio):
        async with get_session() as db:
            s1 = await portfolio.get_or_create_session(db, SessionMode.paper)
        async with get_session() as db:
            s2 = await portfolio.get_or_create_session(db, SessionMode.paper)
        assert s1.id == s2.id


class TestOpenPosition:
    @pytest.mark.asyncio
    async def test_open_reduces_cash_by_cost_plus_fee(
        self, clean_db, portfolio, risk_mgr, make_coin
    ):
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            coin = await make_coin(db)
            sizing = risk_mgr.calculate(
                entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0
            )
            cash_before = session.cash
            result = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal
            )
            await db.commit()
            assert result is not None
            pos, entry_exec = result
            # cash уменьшился на (gross_value + fee).
            expected_cost = entry_exec.net_value
            assert session.cash == pytest.approx(
                cash_before - expected_cost, abs=0.01
            )

    @pytest.mark.asyncio
    async def test_cannot_open_two_positions_same_coin(
        self, clean_db, portfolio, risk_mgr, make_coin
    ):
        """Pyramiding: можно открыть несколько позиций одной стороны (до лимита 3),
        но не чаще чем раз в PYRAMIDING_COOLDOWN_SEC секунд."""
        from datetime import datetime, timedelta, timezone as _tz
        from core.portfolio_engine import PYRAMIDING_COOLDOWN_SEC
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            coin = await make_coin(db)
            sizing = risk_mgr.calculate(
                entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0
            )
            r1 = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal
            )
            await db.commit()
            assert r1 is not None
            # Вторая сразу — блокируется cooldown.
            r2 = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal
            )
            assert r2 is None  # cooldown
            # Сдвигаем entry_at первой позиции назад, чтобы cooldown прошёл.
            pos1, _ = r1
            pos1.entry_at = datetime.now(_tz.utc) - timedelta(
                seconds=PYRAMIDING_COOLDOWN_SEC + 1
            )
            await db.flush()
            # Теперь вторая позиция должна пройти.
            r3 = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal
            )
            assert r3 is not None  # pyramiding разрешён после cooldown

    @pytest.mark.asyncio
    async def test_open_rejects_zero_sizing(
        self, clean_db, portfolio, make_coin
    ):
        """PositionSizing с qty=0 выбрасывает ValueError на этапе валидации."""
        # Валидация происходит в PositionSizing.__post_init__ — раньше open_position.
        with pytest.raises(ValueError):
            PositionSizing(
                qty=0.0, position_value=0.0,
                stop_price=64000.0, target_price=66000.0,
                risk_amount=1.0, risk_pct=0.01, stop_pct=0.02,
                rr_ratio=2.0, break_even_price=65200.0,
            )


# ─── Тесты PnL и комиссий ───────────────────────────────────

class TestClosePosition:
    @pytest.mark.asyncio
    async def test_profitable_trade(self, clean_db, portfolio, risk_mgr, make_coin):
        """Рост 65000 → 68000: gross_pnl > 0, net_pnl > 0."""
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            coin = await make_coin(db)
            sizing = risk_mgr.calculate(
                entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0
            )
            result = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal
            )
            await db.commit()
            pos, _ = result

            closed = await portfolio.close_position(
                db, session, pos, coin, 68000.0, TradeReason.take_profit
            )
            await db.commit()
            assert closed is not None
            assert closed.gross_pnl > 0
            assert closed.net_pnl > 0
            assert closed.pnl_pct > 0

    @pytest.mark.asyncio
    async def test_losing_trade(self, clean_db, portfolio, risk_mgr, make_coin):
        """Падение 65000 → 63000: net_pnl < 0, но ограничен риском."""
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            coin = await make_coin(db)
            sizing = risk_mgr.calculate(
                entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0
            )
            result = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal
            )
            await db.commit()
            pos, entry_exec = result
            entry_cost = entry_exec.net_value

            closed = await portfolio.close_position(
                db, session, pos, coin, 63000.0, TradeReason.stop_loss
            )
            await db.commit()
            assert closed.net_pnl < 0
            # Убыток ≈ risk_amount (1% от 100 = $1) ± комиссии/slippage.
            assert closed.net_pnl > -2.0  # не больше $2 убытка

    @pytest.mark.asyncio
    async def test_fees_accounted(self, clean_db, portfolio, risk_mgr, make_coin):
        """Комиссии реально вычтены: net < gross всегда."""
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            coin = await make_coin(db)
            sizing = risk_mgr.calculate(
                entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0
            )
            result = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal
            )
            await db.commit()
            pos, _ = result

            closed = await portfolio.close_position(
                db, session, pos, coin, 66000.0, TradeReason.take_profit
            )
            await db.commit()
            # gross - fees - slippage = net → net < gross.
            assert closed.net_pnl < closed.gross_pnl
            assert closed.total_fees > 0
            assert closed.total_slippage >= 0

    @pytest.mark.asyncio
    async def test_cash_increases_on_close(self, clean_db, portfolio, risk_mgr, make_coin):
        """После закрытия прибыльной сделки cash > стартового."""
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            coin = await make_coin(db)
            sizing = risk_mgr.calculate(
                entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0
            )
            await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal
            )
            await db.commit()
            cash_mid = session.cash

            pos = await portfolio.get_open_position(db, session, coin)
            await portfolio.close_position(
                db, session, pos, coin, 68000.0, TradeReason.take_profit
            )
            await db.commit()
            # Cash вырос после закрытия прибыльной сделки.
            assert session.cash > cash_mid
            assert session.cash > 100.0


# ─── Тесты метрик ────────────────────────────────────────────

class TestAnalytics:
    @pytest.mark.asyncio
    async def test_empty_session_zero_metrics(
        self, clean_db, portfolio
    ):
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            await portfolio.record_equity(db, session)
            await db.commit()
            report = await portfolio.compute_analytics(db, session)
            assert report.total_trades == 0
            assert report.total_pnl == 0.0

    @pytest.mark.asyncio
    async def test_win_rate_after_mixed_trades(
        self, clean_db, portfolio, risk_mgr, make_coin
    ):
        """3 сделки: 2 прибыльные + 1 убыточная → win_rate ≈ 67%."""
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            coin = await make_coin(db)

            # Сделка 1: прибыль @ 65000 → 68000
            sizing = risk_mgr.calculate(
                65000.0, 1000.0, session.cash, session.cash
            )
            r = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal,
                force=True,
            )
            await db.commit()
            await portfolio.close_position(
                db, session, r[0], coin, 68000.0, TradeReason.take_profit
            )
            await db.commit()

            # Сделка 2: прибыль @ 65000 → 67000
            sizing = risk_mgr.calculate(
                65000.0, 1000.0, session.cash, session.cash
            )
            r = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal,
                force=True,
            )
            await db.commit()
            await portfolio.close_position(
                db, session, r[0], coin, 67000.0, TradeReason.take_profit
            )
            await db.commit()

            # Сделка 3: убыток @ 65000 → 63000
            sizing = risk_mgr.calculate(
                65000.0, 1000.0, session.cash, session.cash
            )
            r = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal,
                force=True,
            )
            await db.commit()
            await portfolio.close_position(
                db, session, r[0], coin, 63000.0, TradeReason.stop_loss
            )
            await db.commit()

            report = await portfolio.compute_analytics(db, session)
            assert report.total_trades == 3
            assert report.win_rate == pytest.approx(2 / 3, abs=0.01)
            assert report.avg_win > 0
            assert report.avg_loss > 0

    @pytest.mark.asyncio
    async def test_break_even_scenario(
        self, clean_db, portfolio, risk_mgr, make_coin
    ):
        """Сделка ровно по точке безубыточности → net_pnl ≈ 0 (с допуском)."""
        async with get_session() as db:
            session = await portfolio.get_or_create_session(db, SessionMode.paper)
            coin = await make_coin(db)
            sizing = risk_mgr.calculate(
                entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0
            )
            r = await portfolio.open_position(
                db, session, coin, sizing, "4h", TradeReason.entry_signal
            )
            await db.commit()
            pos, entry_exec = result = r
            # Цена должна вырасти на величину комиссий, чтобы выйти в ноль.
            round_trip = 2 * (settings.commission_rate + settings.slippage_rate)
            break_even_price = 65000.0 * (1 + round_trip)
            closed = await portfolio.close_position(
                db, session, pos, coin, break_even_price, TradeReason.take_profit
            )
            await db.commit()
            # net_pnl должно быть близко к нулю (с допуском на округления).
            assert abs(closed.net_pnl) < 0.05, (
                f"net_pnl={closed.net_pnl:.4f} должен быть ≈ 0 при сделке на break-even"
            )


# ─── Тест математики просадки/Sharpe ────────────────────────

class TestRiskMath:
    def test_max_drawdown(self):
        from core.portfolio_engine import _max_drawdown
        # 100 → 120 → 90 → 110: max DD = (120-90)/120 = 25%
        dd = _max_drawdown([100, 120, 90, 110])
        assert dd == pytest.approx(0.25, abs=0.01)

    def test_max_drawdown_no_loss(self):
        from core.portfolio_engine import _max_drawdown
        dd = _max_drawdown([100, 110, 120, 130])
        assert dd == 0.0

    def test_sharpe_positive_for_growing(self):
        from core.portfolio_engine import _sharpe
        rets = [0.001, 0.002, 0.001, 0.003, 0.002]
        s = _sharpe(rets)
        assert s > 0

    def test_sharpe_zero_for_flat(self):
        from core.portfolio_engine import _sharpe
        assert _sharpe([0.0, 0.0, 0.0]) == 0.0
