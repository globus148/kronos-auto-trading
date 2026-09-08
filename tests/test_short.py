"""Unit-тесты для short-функционала.

Проверяем:
1. RiskManager.calculate_short: стоп выше входа, target ниже, RR ≥ 2.
2. StrategyEngine._sub_signals_short: высокие при падении, низкие при росте.
3. StrategyEngine.evaluate_short_entry: пропускает при прогнозе на падение.
4. StrategyEngine.evaluate_exit(side="short"): инвертированные условия.
5. PortfolioEngine: полный цикл short (открытие → закрытие → PnL).
"""

from __future__ import annotations

import pytest

from core.indicators import IndicatorSnapshot
from core.risk_manager import PositionSizing, RiskManager
from core.strategy_engine import StrategyEngine
from db.models import TradeReason


# ─── RiskManager.calculate_short ─────────────────────────────


class TestRiskManagerShort:
    def test_short_stop_above_entry(self):
        """Стоп-лосс для short ВЫШЕ цены входа."""
        rm = RiskManager(
            risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0, max_position_pct=0.30,
        )
        sizing = rm.calculate_short(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0,
        )
        assert sizing.stop_price > 65000.0  # стоп выше входа
        assert sizing.target_price < 65000.0  # target ниже входа

    def test_short_rr_ratio(self):
        """RR ratio для short ≥ 2."""
        rm = RiskManager(
            risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0, max_position_pct=0.30,
        )
        sizing = rm.calculate_short(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0,
        )
        assert sizing.rr_ratio >= 2.0

    def test_short_qty_positive(self):
        """qty > 0 и position_value > 0."""
        rm = RiskManager()
        sizing = rm.calculate_short(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0,
        )
        assert sizing.qty > 0
        assert sizing.position_value > 0

    def test_short_break_even_below_entry(self):
        """Цена безубыточности для short НИЖЕ входа (за счёт комиссий)."""
        rm = RiskManager()
        sizing = rm.calculate_short(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0,
        )
        assert sizing.break_even_price < 65000.0

    def test_short_target_from_prediction(self):
        """Target из pred_min_low (ниже входа) — используется с дисконтом."""
        rm = RiskManager(atr_k_stop=1.5, atr_k_take=3.0)
        sizing = rm.calculate_short(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0,
            target_from_prediction=62000.0,  # ниже входа
        )
        # Target между entry и prediction (дисконт 10% к падению).
        expected_target = 65000 - (65000 - 62000) * 0.9
        assert sizing.target_price == pytest.approx(expected_target, abs=1)

    def test_short_max_cash_limit(self):
        """Не вложить больше чем есть cash."""
        rm = RiskManager(risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0, max_position_pct=1.0)
        sizing = rm.calculate_short(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=10.0,
        )
        assert sizing.position_value <= 10.0

    def test_short_atr_clamp_10pct(self):
        """Стоп не глубже 10% от входа."""
        rm = RiskManager(risk_fraction=0.01, atr_k_stop=1.5)
        sizing = rm.calculate_short(
            entry_price=65000.0, atr=20000.0, equity=100.0, cash=100.0,
        )
        stop_pct = (sizing.stop_price - 65000) / 65000
        assert stop_pct <= 0.10


# ─── StrategyEngine short-сигналы ──────────────────────────────


def _make_snap(**overrides) -> IndicatorSnapshot:
    defaults = dict(
        timestamp=1000000, close=65000.0, rsi=55.0,
        macd=10.0, macd_signal=5.0, macd_hist=5.0,
        ema20=64500.0, ema50=64000.0, atr=1000.0,
        adx=25.0, bb_upper=66000.0, bb_lower=63000.0,
    )
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


class TestStrategyEngineShort:
    def test_short_signals_downtrend(self):
        """Нисходящий тренд (EMA20 < EMA50) → f_trend для short высокий."""
        se = StrategyEngine()
        snap = _make_snap(ema20=63500.0, ema50=65500.0)
        result = se.compute_signal(snap)
        assert result.sub_signals_short["f_trend"] > 0.5

    def test_short_signals_uptrend(self):
        """Восходящий → f_trend для short низкий."""
        se = StrategyEngine()
        snap = _make_snap(ema20=66000.0, ema50=64000.0)
        result = se.compute_signal(snap)
        assert result.sub_signals_short["f_trend"] < 0.3

    def test_short_kronos_negative(self):
        """Kronos предсказывает падение → f_kronos для short > 0."""
        se = StrategyEngine()
        snap = _make_snap()
        result = se.compute_signal(
            snap, kronos_pred_return=-0.03, kronos_pred_slope=-0.001,
        )
        assert result.sub_signals_short["f_kronos"] > 0.0

    def test_short_kronos_disabled(self):
        """Без Kronos → f_kronos для short = 0."""
        se = StrategyEngine()
        snap = _make_snap()
        result = se.compute_signal(snap)
        assert result.sub_signals_short["f_kronos"] == 0.0

    def test_short_s_entry_short_range(self):
        """s_entry_short ∈ [0, 1]."""
        se = StrategyEngine()
        for rsi in [30, 45, 55, 70, 85]:
            for ema20, ema50 in [(64000, 66000), (65500, 64500), (66000, 64000)]:
                snap = _make_snap(rsi=rsi, ema20=ema20, ema50=ema50)
                result = se.compute_signal(snap)
                assert 0.0 <= result.s_entry_short <= 1.0, \
                    f"s_entry_short={result.s_entry_short} rsi={rsi} ema20={ema20}"

    def test_long_and_short_exist(self):
        """compute_signal возвращает оба скоринга."""
        se = StrategyEngine()
        snap = _make_snap()
        result = se.compute_signal(snap)
        assert result.s_entry is not None
        assert result.s_entry_short is not None
        assert isinstance(result.sub_signals_short, dict)

    def test_short_momentum_negative_hist(self):
        """MACD histogram < 0 → f_momentum для short высокий."""
        se = StrategyEngine()
        snap = _make_snap(macd_hist=-10.0)
        result = se.compute_signal(snap)
        assert result.sub_signals_short["f_momentum"] > 0.3


class TestEvaluateShortEntry:
    def test_rejects_positive_pred_return(self):
        """Прогноз на рост → short отклонён (pred_return > -min_expected_move)."""
        se = StrategyEngine()
        snap = _make_snap(adx=30.0, ema20=63500.0, ema50=65500.0, rsi=40.0)
        result = se.compute_signal(snap, kronos_pred_return=0.05, kronos_pred_slope=0.001)
        ok, _, _ = se.evaluate_short_entry(
            snap, result.s_entry_short, result.sub_signals_short,
            equity=100.0, cash=100.0,
            kronos_pred_return=0.05,
        )
        assert not ok

    def test_rejects_low_signal(self):
        """s_entry_short < порога → отказ."""
        se = StrategyEngine()
        snap = _make_snap(adx=25.0, ema20=66000.0, ema50=64000.0, rsi=85.0)
        result = se.compute_signal(snap)
        ok, _, _ = se.evaluate_short_entry(
            snap, result.s_entry_short, result.sub_signals_short,
            equity=100.0, cash=100.0,
        )
        assert not ok


class TestEvaluateExitShort:
    def test_short_take_profit(self):
        """Цена упала до target → take_profit для short."""
        se = StrategyEngine()
        snap = _make_snap(close=62000.0)  # ниже target=63000
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=66500.0, target_price=63000.0, bars_held=5,
            side="short",
        )
        assert reason == "take_profit"

    def test_short_stop_loss(self):
        """Цена поднялась до stop → stop_loss для short."""
        se = StrategyEngine()
        snap = _make_snap(close=67000.0)  # выше stop=66500
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=66500.0, target_price=62000.0, bars_held=5,
            side="short",
        )
        assert reason == "stop_loss"

    def test_long_take_profit_unchanged(self):
        """Long take_profit всё ещё работает как раньше (side=long)."""
        se = StrategyEngine()
        snap = _make_snap(close=69000.0)
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=63500.0, target_price=68000.0, bars_held=5,
            side="long",
        )
        assert reason == "take_profit"

    def test_short_signal_reversal_up(self):
        """Разворот вверх (прогноз роста) против short → signal_reversal."""
        se = StrategyEngine()
        snap = _make_snap(close=64000.0)  # между стопом и тейком
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=66500.0, target_price=62000.0, bars_held=5,
            side="short",
            kronos_pred_return=0.03, kronos_pred_slope=0.002,
        )
        assert reason == "signal_reversal"

    def test_short_time_stop(self):
        """Больше max_hold_periods → time_stop."""
        se = StrategyEngine()
        snap = _make_snap(close=64000.0)  # между стопом и тейком
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=66500.0, target_price=62000.0,
            bars_held=50, side="short",
        )
        assert reason == "time_stop"

    def test_default_side_long(self):
        """По умолчанию side=long, старые тесты не ломаются."""
        se = StrategyEngine()
        snap = _make_snap(close=63000.0)
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=63500.0, target_price=68000.0, bars_held=5,
        )
        assert reason == "stop_loss"


# ─── Интеграционный: полный цикл short в PortfolioEngine ────────


class TestPortfolioShort:
    """Полный цикл short в portfolio engine (async, требует БД)."""

    @pytest.fixture
    async def clean_db(self):
        from db.database import init_db, dispose_db, engine
        from sqlalchemy import text

        await init_db()
        async with engine.begin() as conn:
            for t in ["equity_point", "trade", "position", "session", "coin"]:
                await conn.execute(text(f"DELETE FROM {t}"))
        yield
        await dispose_db()

    @pytest.mark.asyncio
    async def test_short_full_cycle_profit(self, clean_db):
        """Short открылся → цена упала → закрытие с прибылью."""
        from db.database import get_session
        from db.models import Coin
        from core.portfolio_engine import PortfolioEngine, get_portfolio
        from core.risk_manager import RiskManager

        pe = PortfolioEngine(commission_rate=0.001, slippage_rate=0.0005)

        async with get_session() as db:
            # Подготовка.
            coin = Coin(symbol="TEST/USDT", default_tf="4h")
            db.add(coin)
            await db.flush()

            session = await pe.get_or_create_session(db)

            # Sizing short.
            rm = RiskManager(
                risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0,
                max_position_pct=0.30, commission_rate=0.001, slippage_rate=0.0005,
            )
            sizing = rm.calculate_short(
                entry_price=65000.0, atr=1000.0,
                equity=session.cash, cash=session.cash,
            )

            # Открыть short.
            result = await pe.open_short_position(
                db, session, coin, sizing, "4h",
            )
            assert result is not None
            pos, execution = result
            assert pos.side.value == "short"

            # Закрыть short при падении цены.
            closed = await pe.close_position(
                db, session, pos, coin, 62000.0,  # цена упала
                reason=TradeReason.manual,
            )
            await db.commit()

            assert closed is not None
            assert closed.net_pnl > 0  # прибыль при падении
            print(f"\n  Short PnL: ${closed.net_pnl:.4f} ({closed.pnl_pct:.2%})")

    @pytest.mark.asyncio
    async def test_short_full_cycle_loss(self, clean_db):
        """Short открылся → цена выросла → закрытие с убытком (stop_loss)."""
        from db.database import get_session
        from db.models import Coin
        from core.portfolio_engine import PortfolioEngine
        from core.risk_manager import RiskManager

        pe = PortfolioEngine(commission_rate=0.001, slippage_rate=0.0005)

        async with get_session() as db:
            coin = Coin(symbol="TEST2/USDT", default_tf="4h")
            db.add(coin)
            await db.flush()

            session = await pe.get_or_create_session(db)

            rm = RiskManager(
                risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0,
                max_position_pct=0.30, commission_rate=0.001, slippage_rate=0.0005,
            )
            sizing = rm.calculate_short(
                entry_price=65000.0, atr=1000.0,
                equity=session.cash, cash=session.cash,
            )

            result = await pe.open_short_position(
                db, session, coin, sizing, "4h",
            )
            assert result is not None
            pos, _ = result

            # Закрыть short при росте цены (убыток).
            closed = await pe.close_position(
                db, session, pos, coin, 67000.0,  # цена выросла
                reason=TradeReason.manual,
            )
            await db.commit()

            assert closed is not None
            assert closed.net_pnl < 0  # убыток при росте
            print(f"\n  Short loss PnL: ${closed.net_pnl:.4f} ({closed.pnl_pct:.2%})")

    @pytest.mark.asyncio
    async def test_cannot_open_long_and_short_same_coin(self, clean_db):
        """Нельзя открыть long и short одновременно по одной монете."""
        from db.database import get_session
        from db.models import Coin
        from core.portfolio_engine import PortfolioEngine
        from core.risk_manager import RiskManager

        pe = PortfolioEngine(commission_rate=0.001, slippage_rate=0.0005)

        async with get_session() as db:
            coin = Coin(symbol="TEST3/USDT", default_tf="4h")
            db.add(coin)
            await db.flush()

            session = await pe.get_or_create_session(db)

            rm = RiskManager(
                risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0,
                max_position_pct=0.30, commission_rate=0.001, slippage_rate=0.0005,
            )

            # Открыть long.
            sizing_long = rm.calculate(
                entry_price=65000.0, atr=1000.0,
                equity=session.cash, cash=session.cash,
            )
            result_long = await pe.open_position(
                db, session, coin, sizing_long, "4h",
            )
            assert result_long is not None

            # Попытка открыть short — должна быть отклонена.
            sizing_short = rm.calculate_short(
                entry_price=65000.0, atr=1000.0,
                equity=session.cash, cash=session.cash,
            )
            result_short = await pe.open_short_position(
                db, session, coin, sizing_short, "4h",
            )
            assert result_short is None  # отклонено
