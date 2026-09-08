"""MarketService — асинхронная обёртка над CCXT для получения данных с Binance.

Тестовый режим использует ТОЛЬКО публичное API (OHLCV, тикер) — ключи не нужны.
Реальный режим (заглушка) потребует ключей, но в прототипе не реализуется.

CCXT в версиях 4.x имеет встроенную async-поддержку (ccxt.async_support).
Обёртка скрывает детали и устойчива к rate-limit-ам через простые повторы.

План: раздел 3 (MarketService), раздел 6 (источник данных OHLCV).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp
import ccxt.async_support as ccxt

from config import settings

logger = logging.getLogger("trading.market")


@dataclass
class Candle:
    """Одна свеча OHLCV. timestamp — миллисекунды epoch (как у CCXT/Binance)."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketService:
    """Асинхронный доступ к бирже через CCXT.

    Синглтон на процесс. Методы безопасны для конкурентных вызовов из торгового
    цикла и роутов.
    """

    def __init__(self) -> None:
        self._exchange: ccxt.Exchange | None = None
        self._session: aiohttp.ClientSession | None = None
        # Простой лок для сериализации сетевых вызовов (защита от rate-limit).
        self._lock = asyncio.Lock()

    # ─── Жизненный цикл ───────────────────────────────────────

    async def _ex(self) -> ccxt.Exchange:
        """Ленивая инициализация обменника.

        ВАЖНО: передаём готовую aiohttp-сессию с ThreadedResolver — на Windows
        дефолтный async-DNS (aiodns/pycares) ломается с ClientConnectorDNSError.
        См. core/http_client.py.
        """
        if self._exchange is None:
            # Ленивый импорт, чтобы модуль импортировался без сети.
            from core.http_client import make_aiohttp_session

            self._session = make_aiohttp_session()
            kwargs: dict[str, Any] = {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                "session": self._session,
            }
            # Ключи подключаем только если они заданы (реальный режим).
            if settings.binance_api_key:
                kwargs["apiKey"] = settings.binance_api_key
                kwargs["secret"] = settings.binance_api_secret
            self._exchange = getattr(ccxt, settings.exchange)(kwargs)
            await self._exchange.load_markets()
            logger.info("MarketService подключён к %s", settings.exchange)
        return self._exchange

    async def close(self) -> None:
        """Корректно закрыть соединение (при остановке приложения)."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
        # CCXT.close() закрывает свою сессию; но мы передали свою — закрываем явно.
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        logger.info("MarketService соединение закрыто")

    # ─── Публичные методы ─────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        since: int | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """Получить до `limit` свечей начиная с `since` (ms epoch).

        CCXT возвращает список [timestamp, o, h, l, c, v].
        """
        ex = await self._ex()
        async with self._lock:
            raw = await ex.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=limit
            )
        return [
            Candle(timestamp=int(r[0]), open=float(r[1]), high=float(r[2]),
                   low=float(r[3]), close=float(r[4]), volume=float(r[5]))
            for r in raw
        ]

    async def fetch_ohlcv_history(
        self,
        symbol: str,
        timeframe: str,
        total: int,
        end_ts_ms: int | None = None,
    ) -> list[Candle]:
        """Скачать `total` свечей вглубь истории пагинацией (план 6.2).

        Binance отдаёт максимум ~1000 свечей за запрос, поэтому идём страницами
        от старых к новым, сдвигая `since`. Удаляем дубли по timestamp.
        """
        ex = await self._ex()
        page = min(1000, total)
        all_candles: dict[int, Candle] = {}

        # Стартовая точка: если end_ts_ms задан — отступаем назад на total свечей.
        tf_ms = ex.parse_timeframe(timeframe) * 1000
        cursor = (end_ts_ms if end_ts_ms else _now_ms()) - total * tf_ms

        async with self._lock:
            while len(all_candles) < total:
                raw = await ex.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=cursor, limit=page
                )
                if not raw:
                    break
                for r in raw:
                    ts = int(r[0])
                    all_candles[ts] = Candle(
                        timestamp=ts, open=float(r[1]), high=float(r[2]),
                        low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                    )
                # Сдвигаем курсор за последнюю полученную свечу.
                cursor = int(raw[-1][0]) + tf_ms
                # Защита от зацикливания, если биржа перестала отдавать новое.
                if len(raw) < page:
                    break

        # Сортировка по времени и обрезка до нужного размера (свежие N).
        candles = sorted(all_candles.values(), key=lambda c: c.timestamp)
        if len(candles) > total:
            candles = candles[-total:]
        logger.info(
            "Скачано %d свечей %s %s (запрошено %d)",
            len(candles), symbol, timeframe, total,
        )
        return candles

    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """Последний тикер (цена, bid/ask, объём 24ч)."""
        ex = await self._ex()
        async with self._lock:
            t = await ex.fetch_ticker(symbol)
        return {
            "last": float(t.get("last") or 0.0),
            "bid": float(t.get("bid") or 0.0),
            "ask": float(t.get("ask") or 0.0),
            "volume": float(t.get("baseVolume") or 0.0),
        }


def _now_ms() -> int:
    """Текущее время в миллисекундах epoch."""
    import time
    return int(time.time() * 1000)


# ─── Глобальный синглтон ──────────────────────────────────────

_market: MarketService | None = None


def get_market() -> MarketService:
    """Синглтон MarketService."""
    global _market
    if _market is None:
        _market = MarketService()
    return _market


# ─── LiveBroker — реальные ордера через кошелёк (live-режим) ──


@dataclass
class LiveOrderResult:
    """Результат исполнения реального рыночного ордера."""
    symbol: str
    side: str               # "buy" | "sell"
    qty: float              # исполненное количество base
    avg_price: float        # средняя цена исполнения
    cost: float             # потрачено/получено quote (qty * avg_price)
    fee: float              # комиссия в quote
    order_id: str           # id ордера на бирже
    raw: dict               # сырой ответ биржи (для отладки)


class LiveBroker:
    """Торговля реальными деньгами через API-ключи кошелька.

    В отличие от MarketService (публичные данные, без ключей), LiveBroker
    создаёт отдельный CCXT-инстанс с ключами конкретного кошелька и ставит
    реальные рыночные ордера. Используется ТОЛЬКО в live-режиме.

    Жизненный цикл: создаётся на операцию, закрывается после. Не синглтон —
    чтобы не держать ключи в памяти дольше необходимого.
    """

    def __init__(self, exchange_name: str, api_key: str, api_secret: str) -> None:
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self._exchange: ccxt.Exchange | None = None
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def _ex(self) -> ccxt.Exchange:
        """Ленивая инициализация обменника с ключами кошелька."""
        if self._exchange is None:
            from core.http_client import make_aiohttp_session

            self._session = make_aiohttp_session()
            self._exchange = getattr(ccxt, self.exchange_name)({
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                "session": self._session,
            })
            await self._exchange.load_markets()
            logger.info(
                "LiveBroker подключён к %s (ключ …%s)",
                self.exchange_name, self.api_key[-4:],
            )
        return self._exchange

    async def close(self) -> None:
        """Закрыть соединение (вызывать после операции)."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_balance(self) -> dict[str, float]:
        """Реальный баланс: {USDT: free_amount, ...}.

        Возвращает словарь {asset: free_balance} для всех активов с ненулевым
        свободным балансом.
        """
        ex = await self._ex()
        async with self._lock:
            balance = await ex.fetch_balance()
        free = balance.get("free", {}) or {}
        return {
            asset: float(amt)
            for asset, amt in free.items()
            if amt and float(amt) > 0
        }

    async def fetch_usdt_balance(self) -> float:
        """Свободный баланс USDT (удобный shorthand)."""
        bal = await self.fetch_balance()
        return float(bal.get("USDT", 0.0))

    async def create_market_buy(self, symbol: str, quote_amount: float) -> LiveOrderResult:
        """Рыночная покупка на `quote_amount` USDT.

        Binance spot: для market-buy можно передать cost (quote) через
        params.quoteOrderQty. CCXT абстрагирует это через create_order с
        type='market' и параметром cost.

        Возвращает результат исполнения.
        """
        ex = await self._ex()
        async with self._lock:
            order = await ex.create_order(
                symbol, type="market", side="buy", amount=None,
                params={"quoteOrderQty": quote_amount},
            )
        return _parse_order(order, symbol, "buy")

    async def create_market_buy_qty(self, symbol: str, qty: float) -> LiveOrderResult:
        """Рыночная покупка точного количества base (qty монет)."""
        ex = await self._ex()
        async with self._lock:
            order = await ex.create_order(
                symbol, type="market", side="buy", amount=qty,
            )
        return _parse_order(order, symbol, "buy")

    async def create_market_sell(self, symbol: str, qty: float) -> LiveOrderResult:
        """Рыночная продажа количества base (qty монет)."""
        ex = await self._ex()
        async with self._lock:
            order = await ex.create_order(
                symbol, type="market", side="sell", amount=qty,
            )
        return _parse_order(order, symbol, "sell")


def _parse_order(order: dict, symbol: str, side: str) -> LiveOrderResult:
    """Извлечь человекочитаемые поля из ответа CCXT create_order.

    CCXT нормализует ответ: filled, average, cost, fee, id.
    """
    filled = float(order.get("filled") or 0.0)
    avg = float(order.get("average") or 0.0)
    cost = float(order.get("cost") or (filled * avg))
    fee_info = order.get("fee") or {}
    fee = float(fee_info.get("cost") or 0.0) if fee_info else 0.0
    order_id = str(order.get("id") or "")
    logger.info(
        "Реальный ордер %s %s: qty=%.8f @ $%.4f cost=$%.4f fee=$%.6f id=%s",
        side.upper(), symbol, filled, avg, cost, fee, order_id,
    )
    return LiveOrderResult(
        symbol=symbol, side=side, qty=filled, avg_price=avg,
        cost=cost, fee=fee, order_id=order_id, raw=order,
    )


async def make_live_broker(wallet) -> LiveBroker:
    """Создать LiveBroker из модели Wallet (с расшифровкой секрета).

    Импорт wallet_service сделан локально, чтобы избежать цикла импортов
    (wallet_service импортирует db.models, который может тянуть core).
    """
    from core.wallet_service import get_decrypted_secret
    return LiveBroker(
        exchange_name=wallet.exchange,
        api_key=wallet.api_key,
        api_secret=get_decrypted_secret(wallet),
    )
