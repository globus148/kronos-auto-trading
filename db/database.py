"""Асинхронная сессия и движок SQLAlchemy.

Использование::

    from db.database import get_session, init_db

    await init_db()                 # создать таблицы (прототип)
    async with get_session() as s:
        ...
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import settings

# Движок создаётся один раз на процесс.Для SQLite нужен connect_args
# (check_same_thread=False), т.к. сессия используется из разных корутин.
_is_sqlite = settings.database_url.startswith("sqlite")

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Контекстный менеджер сессии БД с автоматическим закрытием."""
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """Создать все таблицы. Для прототипа — без Alembic."""
    # Импорт здесь, чтобы модели зарегистрировались в метаданных.
    from db import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

    # Миграция MATIC → POL (Binance переименовал тикер).
    await _migrate_matic_to_pol()

    # Seed: создать записи Coin по умолчанию, если их нет.
    await _seed_default_coins()


async def _migrate_matic_to_pol() -> None:
    """Переименовать MATIC/USDT → POL/USDT (Binance переименовал тикер)."""
    from sqlalchemy import select, update
    from db.models import Coin, Candle as CandleRow

    async with SessionLocal() as session:
        # Проверяем, есть ли MATIC.
        matic = (await session.execute(
            select(Coin).where(Coin.symbol == "MATIC/USDT")
        )).scalar_one_or_none()
        pol = (await session.execute(
            select(Coin).where(Coin.symbol == "POL/USDT")
        )).scalar_one_or_none()

        if matic is not None and pol is None:
            # Переименовываем MATIC → POL.
            matic.symbol = "POL/USDT"
            await session.commit()
            print("  [migrate] MATIC/USDT → POL/USDT")
        elif pol is None:
            # Если нет ни MATIC ни POL — seed создаст POL.
            pass


async def _seed_default_coins() -> None:
    """Создать монеты из settings.default_coins, если ещё не существуют."""
    from sqlalchemy import select
    from db.models import Coin

    async with SessionLocal() as session:
        for symbol in settings.default_coins:
            existing = (await session.execute(
                select(Coin).where(Coin.symbol == symbol)
            )).scalar_one_or_none()
            if existing is None:
                session.add(Coin(symbol=symbol, enabled=True, default_tf="4h"))
                print(f"  [seed] Создана монета: {symbol}")
        await session.commit()


async def dispose_db() -> None:
    """Корректно закрыть пул соединений (вызывать при остановке)."""
    await engine.dispose()
