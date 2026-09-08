"""Unit-тесты для менеджера режима (paper/live)."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from core.mode_manager import ACTIVE_MODE_KEY, get_active_mode, set_active_mode
from db.database import engine, get_session, init_db
from db.models import SessionMode


@pytest.fixture
async def clean_db():
    """Чистая БД."""
    await init_db()
    async with engine.begin() as conn:
        await conn.execute(text(
            f"DELETE FROM strategy_config WHERE key = '{ACTIVE_MODE_KEY}'"
        ))
    yield


@pytest.mark.asyncio
async def test_default_mode_is_paper(clean_db):
    """Без записи в БД — дефолт paper."""
    mode = await get_active_mode()
    assert mode == SessionMode.paper


@pytest.mark.asyncio
async def test_set_and_get_mode(clean_db):
    """Переключение режима: paper → live → paper."""
    mode = await set_active_mode(SessionMode.live)
    assert mode == SessionMode.live
    assert await get_active_mode() == SessionMode.live

    mode = await set_active_mode(SessionMode.paper)
    assert mode == SessionMode.paper
    assert await get_active_mode() == SessionMode.paper


@pytest.mark.asyncio
async def test_set_mode_survives_restart(clean_db):
    """Значение режима сохраняется в БД (перезапуск сессии)."""
    await set_active_mode(SessionMode.live)
    # Симулируем «перезапуск»: читаем снова.
    assert await get_active_mode() == SessionMode.live
