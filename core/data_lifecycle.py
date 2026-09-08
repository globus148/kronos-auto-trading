"""DataLifecycleManager — управление жизненным циклом свечей OHLCV.

Реализует раздел 6 плана:
    6.2  cold start    — первичная загрузка `required_history` свечей
    6.3  warm start    — incremental/полное обновление при перезапуске
    6.4  pruning       — удаление старых «неэффективных» свечей (retention)
    6.5  live update   — добавление 1 свечи за тик + pruning
    6.7  контроль качества — детект дыр, аномалий

Принцип: таблица `candle` всегда = ровное скользящее окно свежайших свечей.
БД не растёт бесконечно.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from core.market_service import Candle, get_market
from db.models import Candle as CandleRow, Coin

logger = logging.getLogger("trading.data")

# Сколько свечей «отставания» считаем нормальным (1 — текущая формируется).
GAP_TOLERANCE = 2


@dataclass
class UpdateReport:
    """Отчёт об одном обновлении пары/ТФ (для логов и UI)."""
    symbol: str
    timeframe: str
    action: str            # "cold_start" | "incremental" | "full_refresh" | "ok"
    fetched: int           # сколько скачано
    inserted: int          # сколько реально вставлено
    pruned: int            # сколько удалено (pruning)
    final_count: int       # итоговое число свечей в БД


class DataLifecycleManager:
    """Скачивание, обновление и очистка свечей по retention-политике."""

    def __init__(self) -> None:
        self._lock_per_pair: dict[str, asyncio.Lock] = {}

    def _pair_lock(self, coin_id: int, tf: str) -> asyncio.Lock:
        key = f"{coin_id}:{tf}"
        if key not in self._lock_per_pair:
            self._lock_per_pair[key] = asyncio.Lock()
        return self._lock_per_pair[key]

    # ─── 6.2 + 6.3: Единая точка входа ───────────────────────

    async def ensure_history(
        self, session: AsyncSession, coin: Coin, timeframe: str
    ) -> UpdateReport:
        """Гарантировать, что для пары/ТФ есть нужная глубина и свежесть.

        Решает сам: cold start, incremental или full refresh (план 6.2/6.3).
        В конце всегда вызывает pruning (план 6.4).
        """
        lock = self._pair_lock(coin.id, timeframe)
        async with lock:
            return await self._ensure_history_locked(session, coin, timeframe)

    async def _ensure_history_locked(
        self, session: AsyncSession, coin: Coin, timeframe: str
    ) -> UpdateReport:
        symbol = coin.symbol
        required = settings.required_history
        tf_ms = self._tf_ms(timeframe)
        now_ms = _now_ms()

        # Текущее состояние в БД.
        last_ts = await self._last_timestamp(session, coin.id, timeframe)
        count = await self._count(session, coin.id, timeframe)

        if last_ts is None or count < required * 0.5:
            # ── 6.2 COLD START: данных нет или критически мало ──
            report = await self._cold_start(session, coin, timeframe, required)
        else:
            gap_ms = now_ms - last_ts
            gap_bars = gap_ms / tf_ms
            if gap_bars <= GAP_TOLERANCE:
                # ── 6.3 OK: данных достаточно, отставание в норме ──
                # Просто incremental: докачать 1-2 последние свечи.
                report = await self._incremental(
                    session, coin, timeframe, since=last_ts + 1, expected=2
                )
                report.action = "ok"
            elif count >= required:
                # ── 6.3 INCREMENTAL: глубина ок, но отстали по свежесте ──
                report = await self._incremental(
                    session, coin, timeframe,
                    since=last_ts + 1, expected=int(gap_bars) + 2,
                )
            else:
                # ── 6.3 FULL REFRESH: большая дыра или фрагментация ──
                # Политика: «лучше свежие и ровные, чем старые с дырами».
                await self._delete_all(session, coin.id, timeframe)
                report = await self._cold_start(session, coin, timeframe, required)
                report.action = "full_refresh"

        # 6.4 — всегда pruning после любого обновления.
        report.pruned = await self.prune(session, coin.id, timeframe)
        report.final_count = await self._count(session, coin.id, timeframe)
        await session.commit()

        logger.info(
            "ensure_history %s %s: action=%s fetched=%d inserted=%d pruned=%d total=%d",
            symbol, timeframe, report.action, report.fetched,
            report.inserted, report.pruned, report.final_count,
        )
        return report

    # ─── 6.2: Cold start ─────────────────────────────────────

    async def _cold_start(
        self, session: AsyncSession, coin: Coin, timeframe: str, total: int
    ) -> UpdateReport:
        """Скачать `total` свечей истории и bulk-insert."""
        market = get_market()
        candles = await market.fetch_ohlcv_history(coin.symbol, timeframe, total)
        inserted = await self._bulk_upsert(session, coin.id, timeframe, candles)
        return UpdateReport(
            symbol=coin.symbol, timeframe=timeframe, action="cold_start",
            fetched=len(candles), inserted=inserted, pruned=0, final_count=0,
        )

    # ─── 6.3: Incremental ────────────────────────────────────

    async def _incremental(
        self, session: AsyncSession, coin: Coin, timeframe: str,
        since: int, expected: int,
    ) -> UpdateReport:
        """Докачать свечи начиная с `since` (ms epoch)."""
        market = get_market()
        candles = await market.fetch_ohlcv(
            coin.symbol, timeframe, since=since, limit=max(expected, 10)
        )
        inserted = await self._bulk_upsert(session, coin.id, timeframe, candles)
        return UpdateReport(
            symbol=coin.symbol, timeframe=timeframe, action="incremental",
            fetched=len(candles), inserted=inserted, pruned=0, final_count=0,
        )

    # ─── 6.5: Live update (одна свеча за тик) ────────────────

    async def update_latest(
        self, session: AsyncSession, coin: Coin, timeframe: str
    ) -> UpdateReport:
        """Добавить последнюю закрытую свечу + pruning (план 6.5).

        Используется торговым циклом на закрытии каждой свечи.
        """
        lock = self._pair_lock(coin.id, timeframe)
        async with lock:
            market = get_market()
            # Берём 2 последние свечи (текущая формируется + последняя закрытая).
            candles = await market.fetch_ohlcv(coin.symbol, timeframe, limit=2)
            inserted = await self._bulk_upsert(session, coin.id, timeframe, candles)
            pruned = await self.prune(session, coin.id, timeframe)
            final = await self._count(session, coin.id, timeframe)
            await session.commit()
            return UpdateReport(
                symbol=coin.symbol, timeframe=timeframe, action="live_update",
                fetched=len(candles), inserted=inserted, pruned=pruned,
                final_count=final,
            )

    # ─── 6.4: Pruning ────────────────────────────────────────

    async def prune(
        self, session: AsyncSession, coin_id: int, timeframe: str
    ) -> int:
        """Удалить свечи старше retention-окна (план 6.4).

        Возвращает число удалённых строк.
        """
        tf_ms = self._tf_ms(timeframe)
        keep_from_ms = _now_ms() - int(settings.retention_size * tf_ms)
        stmt = (
            delete(CandleRow)
            .where(CandleRow.coin_id == coin_id)
            .where(CandleRow.tf == timeframe)
            .where(CandleRow.timestamp < keep_from_ms)
        )
        result = await session.execute(stmt)
        return result.rowcount or 0

    # ─── Чтение свечей из БД ─────────────────────────────────

    async def get_candles(
        self, session: AsyncSession, coin_id: int, timeframe: str, limit: int,
    ) -> list[Candle]:
        """Последние `limit` свечей (свежие) в порядке возрастания времени."""
        stmt = (
            select(CandleRow)
            .where(CandleRow.coin_id == coin_id, CandleRow.tf == timeframe)
            .order_by(CandleRow.timestamp.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()
        # Переворачиваем в хронологическом порядке.
        return [
            Candle(
                timestamp=r.timestamp, open=r.open, high=r.high,
                low=r.low, close=r.close, volume=r.volume,
            )
            for r in reversed(rows)
        ]

    # ─── 6.7: Контроль качества ──────────────────────────────

    async def detect_gaps(
        self, session: AsyncSession, coin_id: int, timeframe: str
    ) -> list[int]:
        """Найти дыры в таймстампах. Возвращает список ожидаемых ts, где нет свечи."""
        candles = await self.get_candles(session, coin_id, timeframe, settings.retention_size)
        if len(candles) < 2:
            return []
        tf_ms = self._tf_ms(timeframe)
        gaps: list[int] = []
        for prev, curr in zip(candles, candles[1:]):
            expected = prev.timestamp + tf_ms
            if curr.timestamp != expected:
                # Считаем сколько свечей пропущено в дыре.
                ts = expected
                while ts < curr.timestamp:
                    gaps.append(ts)
                    ts += tf_ms
        return gaps

    # ─── Вспомогательные ─────────────────────────────────────

    async def _bulk_upsert(
        self, session: AsyncSession, coin_id: int, tf: str, candles: list[Candle],
    ) -> int:
        """Вставка с игнорированием дубликатов (ON CONFLICT DO NOTHING)."""
        if not candles:
            return 0
        rows = [
            {
                "coin_id": coin_id, "tf": tf, "timestamp": c.timestamp,
                "open": c.open, "high": c.high, "low": c.low,
                "close": c.close, "volume": c.volume,
            }
            for c in candles
        ]
        stmt = sqlite_insert(CandleRow).values(rows)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["coin_id", "tf", "timestamp"]
        )
        result = await session.execute(stmt)
        return result.rowcount or 0

    async def _last_timestamp(
        self, session: AsyncSession, coin_id: int, tf: str
    ) -> int | None:
        stmt = (
            select(func.max(CandleRow.timestamp))
            .where(CandleRow.coin_id == coin_id, CandleRow.tf == tf)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _count(
        self, session: AsyncSession, coin_id: int, tf: str
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(CandleRow)
            .where(CandleRow.coin_id == coin_id, CandleRow.tf == tf)
        )
        result = await session.execute(stmt)
        return result.scalar_one()

    async def _delete_all(
        self, session: AsyncSession, coin_id: int, tf: str
    ) -> None:
        stmt = delete(CandleRow).where(
            CandleRow.coin_id == coin_id, CandleRow.tf == tf
        )
        await session.execute(stmt)

    def _tf_ms(self, timeframe: str) -> int:
        """Длительность таймфрейма в миллисекундах."""
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


# ─── Глобальный синглтон ──────────────────────────────────────

_dlm: DataLifecycleManager | None = None


def get_dlm() -> DataLifecycleManager:
    """Синглтон DataLifecycleManager."""
    global _dlm
    if _dlm is None:
        _dlm = DataLifecycleManager()
    return _dlm
