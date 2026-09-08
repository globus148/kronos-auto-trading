"""Синтетические данные OHLCV для офлайн-режима / разработки.

Когда Binance API недоступен (нет сети, песочница, блокировка региона) —
MarketService переключается на генерацию реалистичных свечей.

Модель: геометрическое броуновское движение (GBM) + случайные шоки-волны,
имитирующие тренды и откаты. Достаточно реалистично для отработки индикаторов,
стратегии и UI без живой биржи.

Каждая монета имеет детерминированный seed по символу → одинаковые «рыночные
данные» между запусками (воспроизводимость тестов).
"""

from __future__ import annotations

import hashlib
import math

from core.market_service import Candle


def _seed(symbol: str) -> int:
    """Детерминированный seed из символа."""
    h = hashlib.md5(symbol.encode()).hexdigest()
    return int(h[:8], 16)


def _base_price(symbol: str) -> float:
    """Стартовая цена под символ (чтобы BTC ~ 60000, DOGE ~ 0.1)."""
    seed = _seed(symbol)
    # Логарифмический разброс: от $0.05 до $100000.
    import random
    rng = random.Random(seed)
    log_price = rng.uniform(-3.0, 5.0)  # 10^-3 .. 10^5
    # Известные монеты подгоним ближе к реальности для узнаваемости в UI.
    known = {
        "BTC/USDT": 65000.0, "ETH/USDT": 3500.0, "BNB/USDT": 600.0,
        "SOL/USDT": 150.0, "XRP/USDT": 0.55, "ADA/USDT": 0.45,
        "DOGE/USDT": 0.13, "AVAX/USDT": 35.0, "LINK/USDT": 18.0,
        "POL/USDT": 0.85,
    }
    return known.get(symbol, round(10 ** log_price, 6))


def synth_candles(
    symbol: str,
    timeframe: str,
    total: int,
    end_ts_ms: int | None = None,
    start_price: float | None = None,
) -> list[Candle]:
    """Сгенерировать `total` свечей до `end_ts_ms` (по умолчанию — сейчас).

    Детерминированно по символу: одинаковые данные между запусками, если end_ts
    совпадает. Используется как fallback в MarketService и в тестах.
    """
    import random

    seed = _seed(symbol)
    # Включаем end_ts в seed, чтобы ряды для разных моментов отличались.
    rng = random.Random(seed ^ (end_ts_ms or 0))

    tf_ms = _tf_ms(timeframe)
    end = end_ts_ms or _now_ms()
    # Выравниваем end на границу таймфрейма.
    end = (end // tf_ms) * tf_ms

    price = start_price if start_price else _base_price(symbol)
    # Волатильность за одну свечу (~1.5–3%).
    vol = rng.uniform(0.015, 0.03)
    drift = rng.uniform(-0.0005, 0.001)  # слабый тренд
    candles: list[Candle] = []

    for i in range(total):
        ts = end - (total - 1 - i) * tf_ms
        # Случайные шоки-волны: длиннопериодная синусоида + случайный компонент.
        wave = math.sin(i * 0.07 + seed) * 0.004
        ret = rng.gauss(drift + wave, vol)
        # Редкие сильные движения (5% свечей).
        if rng.random() < 0.05:
            ret += rng.choice([-1, 1]) * rng.uniform(0.02, 0.05)

        o = price
        c = max(o * (1 + ret), 1e-8)
        # Внутрисвечный диапазон.
        span = abs(c - o) / o + rng.uniform(0.002, 0.008)
        h = max(o, c) * (1 + rng.uniform(0, span))
        l = min(o, c) * (1 - rng.uniform(0, span))
        volm = rng.uniform(100, 10000) * (1 + abs(ret) * 20)

        candles.append(Candle(
            timestamp=ts, open=o, high=h, low=l, close=c, volume=volm,
        ))
        price = c

    return candles


def _tf_ms(timeframe: str) -> int:
    import re
    m = re.match(r"^(\d+)([smhdw])$", timeframe)
    if not m:
        raise ValueError(f"Неподдерживаемый таймфрейм: {timeframe}")
    n, unit = int(m.group(1)), m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return n * mult * 1000


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)
