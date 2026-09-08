"""Надёжный aiohttp-клиент для CCXT на Windows.

Проблема: на Windows aiohttp по умолчанию использует `aiodns` (async-DNS на
библиотеке pycares), который не видит системные DNS-серверы и падает с::

    ClientConnectorDNSError: Cannot connect to host ... ssl:default
    [Timeout while contacting DNS servers]

Решение (best practice, см. aiohttp docs + issue #9447):
    1. Явно использовать `ThreadedResolver` — системный getaddrinfo в пуле
       потоков. Не зависит от aiodns/pycares.
    2. Форсировать IPv4 (`socket.AF_INET`) — защита от IPv6-зависаний на
       Windows, где IPv6-резолв может зависать.
    3. Передавать готовую `ClientSession` в CCXT через параметр `session`.

Проверено: api.binance.com/api/v3/ping → 200 за ~0.4с.
"""

from __future__ import annotations

import logging
import socket

import aiohttp

logger = logging.getLogger("trading.http")


def make_aiohttp_session() -> aiohttp.ClientSession:
    """Создать aiohttp-сессию с надёжным резолвером для CCXT.

    Использовать так::

        session = make_aiohttp_session()
        exchange = ccxt.binance({"session": session, ...})
        ...
        await exchange.close()       # закроет и сессию
    """
    connector = aiohttp.TCPConnector(
        resolver=aiohttp.resolver.ThreadedResolver(),
        # Форсируем IPv4: на Windows IPv6-резолв может зависать.
        family=socket.AF_INET,
        force_close=False,
        limit=30,                # максимум соединений в пуле
        limit_per_host=10,
        use_dns_cache=True,
        ttl_dns_cache=300,       # кеш DNS на 5 минут
    )
    timeout = aiohttp.ClientTimeout(total=60, connect=15)
    logger.debug("aiohttp session создана (ThreadedResolver, IPv4)")
    return aiohttp.ClientSession(connector=connector, timeout=timeout)
