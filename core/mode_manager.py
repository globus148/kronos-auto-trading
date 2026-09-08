"""Менеджер активного режима (paper / live).

Хранит текущий активный режим в таблице `strategy_config` (ключ
`active_mode` → "paper" | "live"). Дефолт — paper (безопасность).

Режим определяет, какие сессии использует PortfolioEngine и торгует ли
система реальными деньгами. Все роуты и торговый цикл читают режим через
get_active_mode() вместо захардкоженного SessionMode.paper.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db.database import get_session
from db.models import SessionMode, StrategyConfig

logger = logging.getLogger("trading.mode")

ACTIVE_MODE_KEY = "active_mode"


async def get_active_mode() -> SessionMode:
    """Текущий активный режим (paper по умолчанию)."""
    try:
        async with get_session() as db:
            row = (await db.execute(
                select(StrategyConfig).where(StrategyConfig.key == ACTIVE_MODE_KEY)
            )).scalar_one_or_none()
            if row is not None:
                val = json.loads(row.value_json)
                if val == "live":
                    return SessionMode.live
    except Exception as e:
        logger.debug("get_active_mode: фоллбэк на paper (%s)", e)
    return SessionMode.paper


async def set_active_mode(mode: SessionMode) -> SessionMode:
    """Установить активный режим. Возвращает установленное значение."""
    val = mode.value if isinstance(mode, SessionMode) else str(mode)
    async with get_session() as db:
        stmt = sqlite_insert(StrategyConfig).values(
            key=ACTIVE_MODE_KEY, value_json=json.dumps(val),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[StrategyConfig.key],
            set_={"value_json": stmt.excluded.value_json},
        )
        await db.execute(stmt)
        await db.commit()
    logger.info("Активный режим переключен: %s", val)
    return SessionMode(val)
