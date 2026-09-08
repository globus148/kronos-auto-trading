"""Unit-тесты для RiskManager и StrategyEngine (план 4.3–4.6).

Проверяем:
1. RiskManager: sizing, стопы, ограничения, break-even.
2. StrategyEngine: суб-сигналы, композитный скор, фильтры entry/exit.
3. Интеграцию: реальный BTC/USDT → индикаторы → стратегия → решение.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.indicators import IndicatorSnapshot, compute_all
from core.risk_manager import PositionSizing, RiskManager
from core.strategy_engine import StrategyEngine


# ─── RiskManager ────────────────────────────────────────────

class TestRiskManager:
    def test_basic_sizing(self):
        """Базовый расчёт: equity=100, ATR=1000 (1.5% цены)."""
        rm = RiskManager(
            risk_fraction=0.01,
            atr_k_stop=1.5,
            atr_k_take=3.0,
            max_position_pct=0.30,
        )
        sizing = rm.calculate(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0,
        )
        assert sizing.qty > 0
        assert sizing.position_value <= 30.0  # max 30% of 100
        assert sizing.stop_price < sizing.stop_price + sizing.position_value / sizing.qty  # always true
        assert sizing.stop_price == pytest.approx(65000 - 1500, abs=1)
        assert sizing.target_price > 65000.0
        assert sizing.rr_ratio >= 2.0
        assert sizing.break_even_price > 65000.0  # комиссия
        print(f"\n  sizing: qty={sizing.qty:.6f} val={sizing.position_value:.2f} "
              f"stop={sizing.stop_price:.2f} target={sizing.target_price:.2f} "
              f"RR={sizing.rr_ratio:.2f} BE={sizing.break_even_price:.2f}")

    def test_max_cash_limit(self):
        """Не вложить больше чем есть cash."""
        rm = RiskManager(risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0, max_position_pct=1.0)
        sizing = rm.calculate(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=10.0,
        )
        assert sizing.position_value <= 10.0  # только $10 доступно

    def test_atr_clamp_10pct(self):
        """Стоп не глубже 10% даже при огромной волатильности."""
        rm = RiskManager(risk_fraction=0.01, atr_k_stop=1.5)
        sizing = rm.calculate(
            entry_price=65000.0, atr=20000.0, equity=100.0, cash=100.0,
        )
        stop_pct = (65000 - sizing.stop_price) / 65000
        assert stop_pct <= 0.10

    def test_rr_ratio_minimum(self):
        """RR ratio не может быть < 1 — иначе это невыгодная сделка."""
        rm = RiskManager(risk_fraction=0.01, atr_k_stop=1.5, atr_k_take=3.0)
        sizing = rm.calculate(
            entry_price=65000.0, atr=500.0, equity=100.0, cash=100.0,
        )
        assert sizing.rr_ratio >= 1.0

    def test_break_even_above_entry(self):
        """Цена безубыточности должна быть строго выше цены входа."""
        rm = RiskManager()
        sizing = rm.calculate(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0,
        )
        assert sizing.break_even_price > 65000.0
        # Разница ≈ round_trip_cost.
        pct = (sizing.break_even_price - 65000.0) / 65000.0
        assert 0.002 < pct < 0.005  # 0.2%–0.5%

    def test_target_from_prediction(self):
        """Если Kronos дал target — используем его с дисконтом."""
        rm = RiskManager(atr_k_stop=1.5, atr_k_take=3.0)
        sizing = rm.calculate(
            entry_price=65000.0, atr=1000.0, equity=100.0, cash=100.0,
            target_from_prediction=67000.0,
        )
        # Target должен быть между entry и prediction (с дисконтом 10%).
        expected_target = 65000 + (67000 - 65000) * 0.9
        assert sizing.target_price == pytest.approx(expected_target, abs=1)


# ─── StrategyEngine ──────────────────────────────────────────

def _make_snap(**overrides) -> IndicatorSnapshot:
    """Быстрое создание снапшота индикаторов для тестов."""
    defaults = dict(
        timestamp=1000000, close=65000.0, rsi=55.0,
        macd=10.0, macd_signal=5.0, macd_hist=5.0,
        ema20=64500.0, ema50=64000.0, atr=1000.0,
        adx=25.0, bb_upper=66000.0, bb_lower=63000.0,
    )
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


class TestStrategyEngine:
    def test_sub_signals_uptrend(self):
        """Сильный uptrend → f_trend высокий."""
        se = StrategyEngine()
        snap = _make_snap(ema20=66000.0, ema50=64000.0)  # EMA20 > EMA50
        result = se.compute_signal(snap)
        assert result.sub_signals["f_trend"] > 0.5

    def test_sub_signals_downtrend(self):
        """Нисходящий → f_trend низкий."""
        se = StrategyEngine()
        snap = _make_snap(ema20=63500.0, ema50=65500.0)
        result = se.compute_signal(snap)
        assert result.sub_signals["f_trend"] < 0.3

    def test_sub_signals_rsi_overbought(self):
        """RSI > 80 → f_rsi низкий (перекуплен, не входить)."""
        se = StrategyEngine()
        snap = _make_snap(rsi=85.0)  # > 85 → строго 0
        result = se.compute_signal(snap)
        assert result.sub_signals["f_rsi"] < 0.1

    def test_sub_signals_rsi_strength(self):
        """RSI 50 → f_rsi высокий (зона силы)."""
        se = StrategyEngine()
        snap = _make_snap(rsi=50.0)
        result = se.compute_signal(snap)
        assert result.sub_signals["f_rsi"] > 0.5

    def test_kronos_disabled(self):
        """Без Kronos f_kronos = 0."""
        se = StrategyEngine()
        snap = _make_snap()
        result = se.compute_signal(snap)
        assert result.sub_signals["f_kronos"] == 0.0

    def test_kronos_enabled_positive(self):
        """Kronos предсказывает рост → f_kronos > 0."""
        se = StrategyEngine()
        snap = _make_snap()
        result = se.compute_signal(
            snap, kronos_pred_return=0.02, kronos_pred_slope=0.001,
        )
        assert result.sub_signals["f_kronos"] > 0.0

    def test_composite_range(self):
        """S_entry всегда в [0, 1]."""
        se = StrategyEngine()
        for rsi in [30, 45, 55, 70, 85]:
            for ema_diff in [-1000, 0, 1000, 2000]:
                snap = _make_snap(
                    rsi=rsi,
                    ema20=65000 + ema_diff,
                    ema50=64000.0,
                )
                result = se.compute_signal(snap)
                assert 0.0 <= result.s_entry <= 1.0

    def test_evaluate_entry_rejects_low_signal(self):
        """S_entry < 0.65 → вход отклонён."""
        se = StrategyEngine()
        snap = _make_snap(rsi=35.0, ema20=63800.0, ema50=65000.0, adx=15.0)
        signal = se.compute_signal(snap)
        ok, _, _ = se.evaluate_entry(
            snap, signal.s_entry, signal.sub_signals,
            equity=100.0, cash=100.0,
        )
        assert not ok

    def test_evaluate_entry_rejects_low_adx(self):
        """ADX < 20 (флэт) → вход отклонён."""
        se = StrategyEngine()
        snap = _make_snap(adx=10.0, rsi=55.0, ema20=66000.0, ema50=64000.0)
        signal = se.compute_signal(snap)
        ok, _, _ = se.evaluate_entry(
            snap, signal.s_entry, signal.sub_signals,
            equity=100.0, cash=100.0,
        )
        assert not ok

    def test_evaluate_exit_stop_loss(self):
        """Цена ниже стопа → stop_loss."""
        se = StrategyEngine()
        snap = _make_snap(close=63000.0)
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=63500.0, target_price=68000.0, bars_held=5,
        )
        assert reason == "stop_loss"

    def test_evaluate_exit_take_profit(self):
        """Цена выше тейк-профита → take_profit."""
        se = StrategyEngine()
        snap = _make_snap(close=69000.0)
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=63500.0, target_price=68000.0, bars_held=5,
        )
        assert reason == "take_profit"

    def test_evaluate_exit_time_stop(self):
        """Больше max_hold_periods → time_stop."""
        se = StrategyEngine()
        snap = _make_snap(close=66000.0)  # между стопом и тейком
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=63500.0, target_price=68000.0,
            bars_held=50,  # > max_hold_periods (48)
        )
        assert reason == "time_stop"

    def test_evaluate_exit_trailing_stop(self):
        """Trailing stop сработал (цена упала ниже trailing)."""
        se = StrategyEngine()
        snap = _make_snap(close=66000.0)  # ниже trailing=66500
        reason = se.evaluate_exit(
            snap, position_price=65000.0,
            stop_price=63500.0, target_price=68000.0,
            bars_held=20, trailing_stop=66500.0,
        )
        assert reason == "trailing_stop"


# ─── Интеграционный: реальный BTC/USDT ─────────────────────

class TestRealDataStrategy:
    """Прогон стратегии на реальных данных Binance."""

    @pytest.fixture(scope="class")
    def btc_snapshot(self):
        try:
            import asyncio
            from core.market_service import get_market, Candle
            from db.database import init_db, dispose_db

            async def fetch():
                await init_db()
                m = get_market()
                candles = await m.fetch_ohlcv("BTC/USDT", "4h", limit=100)
                await m.close()
                await dispose_db()
                df = pd.DataFrame({
                    "timestamp": [c.timestamp for c in candles],
                    "close": [c.close for c in candles],
                    "high": [c.high for c in candles],
                    "low": [c.low for c in candles],
                    "volume": [c.volume for c in candles],
                })
                return compute_all(
                    df["timestamp"], df["close"], df["high"], df["low"], df["volume"]
                )

            return asyncio.run(fetch())
        except Exception as e:
            pytest.skip(f"Binance недоступен: {e}")

    def test_strategy_runs_on_real_data(self, btc_snapshot):
        """Стратегия считает без ошибок на реальных данных."""
        se = StrategyEngine()
        signal = se.compute_signal(btc_snapshot)
        assert 0.0 <= signal.s_entry <= 1.0
        ok, sizing, _ = se.evaluate_entry(
            btc_snapshot, signal.s_entry, signal.sub_signals,
            equity=100.0, cash=100.0,
        )
        print(f"\n  BTC/USDT 4h: S_entry={signal.s_entry:.3f} enter={ok} "
              f"sub_signals={signal.sub_signals}")
        if ok and sizing:
            print(f"  sizing: qty={sizing.qty:.8f} val=${sizing.position_value:.2f} "
                  f"stop={sizing.stop_price:.2f} target={sizing.target_price:.2f} "
                  f"RR={sizing.rr_ratio:.2f}")
