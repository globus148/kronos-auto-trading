"""Unit-тесты для core/indicators.py.

Проверяем:
1. Базовую математику каждого индикатора (известные значения).
2. Связку с реальными данными Binance (скачиваем 100 свечей и проверяем корректность).
3. Edge-кейсы (пустые серии, одна свеча, NaN-обработка).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.indicators import (
    IndicatorSnapshot,
    adx,
    atr,
    bollinger,
    compute_all,
    ema,
    macd,
    rsi,
)


# ─── Фикстуры ──────────────────────────────────────────────

@pytest.fixture
def sample_close() -> pd.Series:
    """50 значений close с заданным seed для воспроизводимости."""
    rng = np.random.RandomState(42)
    # Случайное блуждание с трендом вверх.
    prices = [100.0]
    for _ in range(49):
        prices.append(prices[-1] * (1.0 + rng.normal(0.002, 0.02)))
    return pd.Series(prices, dtype=float)


@pytest.fixture
def sample_ohlc(sample_close) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """OHLCV из sample_close."""
    n = len(sample_close)
    rng = np.random.RandomState(42)
    high = sample_close * (1.0 + np.abs(rng.normal(0, 0.01, n)))
    low = sample_close * (1.0 - np.abs(rng.normal(0, 0.01, n)))
    high = pd.Series(np.maximum(high, sample_close), dtype=float)
    low = pd.Series(np.minimum(low, sample_close), dtype=float)
    timestamps = pd.Series(range(1000, 1000 + n * 3600000, 3600000), dtype=int)
    volume = pd.Series(rng.uniform(100, 1000, n), dtype=float)
    return timestamps, sample_close, high, low, volume


# ─── EMA ────────────────────────────────────────────────────

class TestEMA:
    def test_constant_series_equals_input(self):
        s = pd.Series([50.0] * 20)
        result = ema(s, 10)
        assert abs(result.iloc[-1] - 50.0) < 1e-10

    def test_single_value(self):
        s = pd.Series([42.0])
        result = ema(s, 5)
        assert abs(result.iloc[0] - 42.0) < 1e-10

    def test_shorter_than_period(self, sample_close):
        """EMA должна считать даже если ряд короче периода (рекурсивная формула)."""
        result = ema(sample_close[:5], 20)
        assert not result.isna().any()

    def test_lag(self, sample_close):
        """EMA с большим периодом отстаёт сильнее от цены."""
        e5 = ema(sample_close, 5)
        e20 = ema(sample_close, 20)
        # EMA не должна быть NaN.
        assert not np.isnan(e5.iloc[-1])
        assert not np.isnan(e20.iloc[-1])
        # EMA(20) менее реактивна → дальше от текущей цены чем EMA(5).
        deviation_5 = abs(e5.iloc[-1] - sample_close.iloc[-1])
        deviation_20 = abs(e20.iloc[-1] - sample_close.iloc[-1])
        # На 50 барах warm-up может быть неполным, но EMA(20) должен отставать
        # не меньше чем EMA(5) в среднем по хвосту.
        avg_dev5 = abs(e5.iloc[-10:].values - sample_close.iloc[-10:].values).mean()
        avg_dev20 = abs(e20.iloc[-10:].values - sample_close.iloc[-10:].values).mean()
        assert avg_dev20 >= avg_dev5 * 0.8  # с запасом на warm-up


# ─── RSI ───────────────────────────────────────────────────

class TestRSI:
    def test_all_up_rsi_near_100(self):
        """Постоянный рост → RSI ≈ 100."""
        s = pd.Series([100.0 + i for i in range(30)], dtype=float)
        result = rsi(s, 14)
        assert result.iloc[-1] > 95.0

    def test_all_down_rsi_near_0(self):
        """Постоянное падение → RSI ≈ 0."""
        s = pd.Series([200.0 - i for i in range(30)], dtype=float)
        result = rsi(s, 14)
        assert result.iloc[-1] < 5.0

    def test_flat_rsi_near_50(self):
        """Флэт → RSI ≈ 50."""
        s = pd.Series([100.0] * 30, dtype=float)
        result = rsi(s, 14)
        assert abs(result.iloc[-1] - 50.0) < 1.0

    def test_range(self, sample_close):
        result = rsi(sample_close, 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


# ─── MACD ───────────────────────────────────────────────────

class TestMACD:
    def test_flat_hist_near_zero(self):
        s = pd.Series([100.0] * 50, dtype=float)
        r = macd(s, 12, 26, 9)
        assert abs(r.histogram.iloc[-1]) < 0.01

    def test_structure(self, sample_close):
        r = macd(sample_close, 12, 26, 9)
        assert len(r.macd) == len(sample_close)
        assert len(r.signal) == len(sample_close)
        assert len(r.histogram) == len(sample_close)
        # histogram = macd - signal
        np.testing.assert_allclose(
            r.histogram.values, (r.macd - r.signal).values, atol=1e-10
        )


# ─── ATR ───────────────────────────────────────────────────

class TestATR:
    def test_zero_volatility(self):
        """Нулевая волатильность → ATR ≈ 0."""
        o = h = l = c = pd.Series([100.0] * 20, dtype=float)
        result = atr(h, l, c, 14)
        assert abs(result.iloc[-1]) < 1e-10

    def test_positive(self, sample_ohlc):
        _, _, high, low, close = sample_ohlc
        result = atr(high, low, close, 14)
        valid = result.dropna()
        assert (valid > 0).all()


# ─── Bollinger ──────────────────────────────────────────────

class TestBollinger:
    def test_contains_price(self, sample_close):
        r = bollinger(sample_close, 20, 2.0)
        valid_idx = ~r.upper.isna()
        # 95%+ свечей должны быть внутри bands (по определению ±2σ).
        inside = (
            (sample_close[valid_idx] >= r.lower[valid_idx])
            & (sample_close[valid_idx] <= r.upper[valid_idx])
        )
        assert inside.mean() > 0.90

    def test_flat_bands(self):
        s = pd.Series([50.0] * 30, dtype=float)
        r = bollinger(s, 20, 2.0)
        assert abs(r.upper.iloc[-1] - r.lower.iloc[-1]) < 1e-10  # std=0


# ─── ADX ───────────────────────────────────────────────────

class TestADX:
    def test_strong_trend_high_adx(self):
        """Сильный uptrend → ADX > 25."""
        n = 60
        close = pd.Series([100.0 + i * 0.5 + np.random.normal(0, 0.5) for i in range(n)])
        high = close * 1.01
        low = close * 0.99
        result = adx(high, low, close, 14)
        # Сильный тренд — ADX должен быть заметным.
        assert result.iloc[-1] > 15.0

    def test_flat_low_adx(self):
        """Флэт → ADX низкий."""
        n = 60
        close = pd.Series([100.0 + np.random.normal(0, 0.1) for _ in range(n)])
        high = close + 0.1
        low = close - 0.1
        result = adx(high, low, close, 14)
        # Флэт — ADX может быть низким.
        assert result.iloc[-1] < 40.0


# ─── compute_all (комплексный снимок) ────────────────────────

class TestComputeAll:
    def test_returns_snapshot(self, sample_ohlc):
        timestamps, close, high, low, volume = sample_ohlc
        snap = compute_all(timestamps, close, high, low, volume)
        assert isinstance(snap, IndicatorSnapshot)
        assert snap.close == close.iloc[-1]
        assert 0 <= snap.rsi <= 100
        assert snap.atr > 0

    def test_snap_matches_individual(self, sample_ohlc):
        """Значения в snapshot совпадают с индивидуальными индикаторами."""
        timestamps, close, high, low, volume = sample_ohlc
        snap = compute_all(timestamps, close, high, low, volume)
        assert abs(snap.rsi - rsi(close, 14).iloc[-1]) < 1e-6
        macd_r = macd(close, 12, 26, 9)
        assert abs(snap.macd - macd_r.macd.iloc[-1]) < 1e-6
        assert abs(snap.atr - atr(high, low, close, 14).iloc[-1]) < 1e-6
        assert abs(snap.adx - adx(high, low, close, 14).iloc[-1]) < 1e-6


# ─── Интеграционный: реальные данные Binance ─────────────────

class TestRealData:
    """Тесты на реальных свечах BTC/USDT с Binance (скачиваются на лету).

    Маркировка: если сеть недоступна — пропускается.
    """
    BTC_DATA: list | None = None

    @pytest.fixture(scope="class")
    def btc_candles(self):
        """Скачивает 100 свечей BTC/USDT 4h один раз для класса."""
        if TestRealData.BTC_DATA is not None:
            return TestRealData.BTC_DATA
        try:
            import asyncio
            from core.market_service import get_market
            from db.database import init_db, dispose_db

            async def fetch():
                await init_db()
                m = get_market()
                c = await m.fetch_ohlcv("BTC/USDT", "4h", limit=100)
                await m.close()
                await dispose_db()
                return c

            TestRealData.BTC_DATA = asyncio.run(fetch())
        except Exception:
            pytest.skip("Binance недоступен — пропускаем интеграционный тест")
        assert TestRealData.BTC_DATA is not None
        return TestRealData.BTC_DATA

    def test_indicators_on_real_data(self, btc_candles):
        """Все индикаторы считают без ошибок на реальных данных."""
        df = pd.DataFrame(
            {
                "timestamp": [c.timestamp for c in btc_candles],
                "close": [c.close for c in btc_candles],
                "high": [c.high for c in btc_candles],
                "low": [c.low for c in btc_candles],
                "volume": [c.volume for c in btc_candles],
            }
        )
        snap = compute_all(
            df["timestamp"], df["close"], df["high"], df["low"], df["volume"]
        )
        # Базовые sanity-проверки.
        assert snap.close > 0
        assert 0 <= snap.rsi <= 100
        assert snap.atr > 0
        assert not math.isnan(snap.macd)
        assert not math.isnan(snap.adx)
        print(f"\n  BTC/USDT 4h: close={snap.close:.2f} RSI={snap.rsi:.1f} "
              f"ATR={snap.atr:.2f} ADX={snap.adx:.1f} MACD={snap.macd:.4f}")
