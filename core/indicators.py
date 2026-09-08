"""Технические индикаторы для сигнального слоя (план, раздел 4.2).

Все индикаторы реализованы на pandas/numpy без внешних зависимостей.
Принимают pd.Series (обычно close-цены) и возвращают pd.Series.

Используется StrategyEngine для расчёта feature vector и композитного сигнала.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ─── EMA (Exponential Moving Average) ──────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Экспоненциальная скользящая средняя."""
    return series.ewm(span=period, adjust=False).mean()


# ─── RSI (Relative Strength Index) ─────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI по методу Wilder (сглаженный).

    Значения ∈ [0, 100]. >70 — перекуплен, <30 — перепродан.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder-сглаженный RSI: используем min_periods=1, чтобы значения
    # считались с первого бара (ранее min_periods=period давал NaN на первых bar).
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=1, adjust=False).mean()

    # CLAMP: avg_loss=0 → rs=+∞ (RSI=100); avg_loss≈0 → cap rs чтобы RSI≈100.
    # avg_gain=0, avg_loss=0 → rs=0 (RSI=50).
    avg_loss_safe = avg_loss.where(avg_loss > 1e-12, 1e-12)
    rs = avg_gain / avg_loss_safe

    result = 100.0 - (100.0 / (1.0 + rs))
    # Флэт (оба gain/loss ≈ 0) → 50.
    flat_mask = (avg_gain < 1e-12) & (avg_loss < 1e-12)
    result = result.where(~flat_mask, 50.0)
    return result


# ─── MACD (Moving Average Convergence Divergence) ──────────

@dataclass
class MACDResult:
    """Результат MACD: три серии (линия, сигнал, гистограмма)."""
    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> MACDResult:
    """MACD = EMA(fast) − EMA(slow). Signal = EMA(MACD, signal_period).
    Histogram = MACD − Signal.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return MACDResult(macd=macd_line, signal=signal_line, histogram=histogram)


# ─── ATR (Average True Range) ───────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder-сглаженный).

    Измеряет волатильность. Используется для стоп-лоссов и размер позиции.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ─── Bollinger Bands ───────────────────────────────────────

@dataclass
class BollingerResult:
    """Результат Bollinger Bands."""
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series


def bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerResult:
    """Bollinger Bands: SMA ± num_std × stddev."""
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return BollingerResult(upper=upper, middle=middle, lower=lower)


# ─── ADX (Average Directional Index) ──────────────────────

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index (Wilder-сглаженный).

    Измеряет силу тренда независимо от направления.
    <20 — флэт/без тренда, >25 — тренд.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low
    # Нули где движение отрицательное.
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = atr(high, low, close, period=1)  # True Range (несглаженный)
    atr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * (
        plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        / atr_smooth.replace(0, np.nan)
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        / atr_smooth.replace(0, np.nan)
    )

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx_line


# ─── Комплексный снимок всех индикаторов ────────────────────

@dataclass
class IndicatorSnapshot:
    """Снимок всех индикаторов на текущем баре (последнее значение каждой серии).

    Используется StrategyEngine для расчёта S_entry/S_exit.
    """
    timestamp: int       # ms epoch
    close: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    ema20: float
    ema50: float
    atr: float
    adx: float
    bb_upper: float
    bb_lower: float


def compute_all(
    timestamps: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> IndicatorSnapshot:
    """Посчитать все индикаторы и вернуть снимок на последнем баре.

    На вход — полный DataFrame (как минимум indicator_warmup + some).
    На выход — один IndicatorSnapshot (последние значения каждой серии).
    """
    rsi_val = rsi(close, 14).iloc[-1]
    macd_r = macd(close, 12, 26, 9)
    atr_val = atr(high, low, close, 14).iloc[-1]
    adx_val = adx(high, low, close, 14).iloc[-1]
    bb_r = bollinger(close, 20, 2.0)

    return IndicatorSnapshot(
        timestamp=int(timestamps.iloc[-1]),
        close=float(close.iloc[-1]),
        rsi=float(rsi_val),
        macd=float(macd_r.macd.iloc[-1]),
        macd_signal=float(macd_r.signal.iloc[-1]),
        macd_hist=float(macd_r.histogram.iloc[-1]),
        ema20=float(ema(close, 20).iloc[-1]),
        ema50=float(ema(close, 50).iloc[-1]),
        atr=float(atr_val),
        adx=float(adx_val),
        bb_upper=float(bb_r.upper.iloc[-1]),
        bb_lower=float(bb_r.lower.iloc[-1]),
    )
