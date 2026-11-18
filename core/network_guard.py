"""NetworkGuard — пауза торговли, пока нет связи с биржей.

Когда пропадает интернет, каждый запрос к бирже висит до таймаута и падает
с длинным traceback'ом. Раньше бот всё равно проходил все монеты (19 × 2 ТФ
каждые 15 минут), и за сутки без сети в trading.log набегало ~6000 строк
ошибок. Стопы и тейки при этом всё равно не проверялись — цен нет.

Теперь:
    - перед каждым прогоном цикл спрашивает ensure_online();
    - первая сетевая ошибка внутри прогона (report_failure) переводит бота
      в режим «нет связи», и оставшиеся монеты не трогаются;
    - пока связи нет, торговля стоит; раз в минуту идёт короткая проверка
      (probe), в лог ничего не пишется;
    - когда связь вернулась — одна строка в лог с длительностью простоя,
      и цикл сразу делает внеочередной прогон (проверить стопы/тейки).

В логе за весь обрыв остаются две строки: «НЕТ СВЯЗИ…» и «Связь восстановлена…».
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time

logger = logging.getLogger("trading.network")

# Сколько ждём ответа биржи при проверке связи.
PROBE_TIMEOUT_SEC = 10.0


def is_network_error(exc: BaseException) -> bool:
    """Ошибка связи (нет интернета / DNS / таймаут), а не логики программы."""
    try:
        import ccxt  # noqa: WPS433

        if isinstance(exc, ccxt.NetworkError):  # RequestTimeout, ExchangeNotAvailable…
            return True
    except Exception:  # pragma: no cover — ccxt есть всегда, но не падаем без него
        pass
    try:
        import aiohttp

        if isinstance(exc, aiohttp.ClientError):  # ClientConnectorDNSError и т.п.
            return True
    except Exception:  # pragma: no cover
        pass
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, socket.gaierror))


def short_reason(exc: BaseException) -> str:
    """Короткое описание причины для лога (без URL и traceback)."""
    cause = exc
    # ccxt заворачивает исходную ошибку — ищем самую информативную.
    seen = 0
    while cause.__cause__ is not None and seen < 5:
        cause = cause.__cause__
        seen += 1
    text = str(cause).strip() or type(cause).__name__
    if "getaddrinfo" in text or isinstance(cause, socket.gaierror):
        return "DNS не отвечает (нет интернета?)"
    if isinstance(cause, (asyncio.TimeoutError, TimeoutError)) or "timed out" in text.lower():
        return "таймаут запроса к бирже"
    return f"{type(cause).__name__}: {text[:120]}"


class NetworkGuard:
    """Состояние связи с биржей: online / offline."""

    def __init__(self, probe=None) -> None:
        # probe — корутина без аргументов: успешно вернулась = связь есть.
        self._probe = probe
        self.online = True
        self.offline_since: float | None = None   # time.time() начала обрыва
        self.last_reason = ""
        self._lock = asyncio.Lock()

    async def _default_probe(self) -> None:
        from core.market_service import get_market

        await get_market().ping()

    async def probe(self) -> bool:
        """Проверить связь с биржей прямо сейчас."""
        probe = self._probe or self._default_probe
        try:
            await asyncio.wait_for(probe(), timeout=PROBE_TIMEOUT_SEC)
            return True
        except Exception as exc:
            self.last_reason = short_reason(exc)
            return False

    def report_failure(self, exc: BaseException) -> None:
        """Сетевая ошибка во время работы → перейти в режим «нет связи»."""
        self.last_reason = short_reason(exc)
        self._go_offline()

    def _go_offline(self) -> None:
        if not self.online:
            return
        self.online = False
        self.offline_since = time.time()
        logger.warning(
            "НЕТ СВЯЗИ с биржей (%s). Торговля приостановлена: сделки не "
            "открываются, стопы и тейки не проверяются. Проверка связи раз в минуту.",
            self.last_reason,
        )

    def _go_online(self) -> float:
        """Вернуть «связь есть». Возвращает длительность простоя в секундах."""
        downtime = time.time() - (self.offline_since or time.time())
        self.online = True
        self.offline_since = None
        logger.info("Связь с биржей восстановлена (простой %s). Торговля продолжается.",
                    _fmt_duration(downtime))
        return downtime

    async def ensure_online(self) -> bool:
        """Можно ли сейчас работать с биржей.

        Если связь была — сразу True (без лишнего запроса: обрыв заметим по
        первой ошибке). Если связи нет — пробуем один раз.
        """
        if self.online:
            return True
        async with self._lock:
            if self.online:
                return True
            if await self.probe():
                self._go_online()
                return True
            return False

    async def check(self) -> bool:
        """Периодическая проверка (раз в минуту). True — если связь вернулась."""
        if self.online:
            return False
        async with self._lock:
            if self.online:
                return False
            if await self.probe():
                self._go_online()
                return True
            return False

    def status(self) -> dict:
        return {
            "online": self.online,
            "offline_since": (
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.offline_since))
                if self.offline_since else None
            ),
            "offline_seconds": int(time.time() - self.offline_since) if self.offline_since else 0,
            "reason": "" if self.online else self.last_reason,
        }


def _fmt_duration(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} ч {m} мин"
    if m:
        return f"{m} мин {s} с"
    return f"{s} с"


class DropNetworkNoise(logging.Filter):
    """Убирает из лога дубли от aiohttp при обрыве сети.

    asyncio пишет «gaierror exception in shielded future» с полным traceback'ом
    на каждый неудачный DNS-запрос — это та же ошибка, что мы уже поймали
    и записали одной строкой. Остальные сообщения asyncio не трогаем.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and record.exc_info[1] is not None:
            if is_network_error(record.exc_info[1]):
                return False
        msg = record.getMessage()
        if "exception in shielded future" in msg and "gaierror" in msg:
            return False
        return True


_guard: NetworkGuard | None = None


def get_network_guard() -> NetworkGuard:
    global _guard
    if _guard is None:
        _guard = NetworkGuard()
    return _guard
