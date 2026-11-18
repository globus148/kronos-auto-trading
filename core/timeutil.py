"""Единый формат времени проекта: UTC.

Всё, что проект пишет сам — логи (trading.log, trades.log), signals.csv,
поля DateTime в БД, расписание APScheduler — ведётся в UTC, как и свечи
биржи (timestamp в ms epoch = UTC). Локальное время (МСК) нигде не
используется, чтобы записи из разных источников сверялись без пересчёта.

Текстовый формат везде один: ``YYYY-MM-DD HH:MM:SS`` (см. TIME_FORMAT).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

# Единый текстовый формат времени (логи, CSV, UI).
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Таймзона планировщика (cron-задачи привязаны к закрытию свечей биржи в UTC).
SCHEDULER_TZ = "UTC"


def utcnow() -> datetime:
    """Текущее время UTC как naive datetime (так хранит DateTime в SQLite).

    Замена устаревшему datetime.utcnow().
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_str(dt: datetime | None = None) -> str:
    """Время в едином формате ``YYYY-MM-DD HH:MM:SS`` (UTC).

    aware-datetime приводится к UTC; naive считается уже UTC.
    """
    if dt is None:
        dt = utcnow()
    elif dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime(TIME_FORMAT)


def ms_to_utc_str(ts_ms: int) -> str:
    """ms epoch (timestamp свечи) → ``YYYY-MM-DD HH:MM:SS`` UTC."""
    return time.strftime(TIME_FORMAT, time.gmtime(ts_ms / 1000))
