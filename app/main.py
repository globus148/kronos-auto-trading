"""Точка входа FastAPI-приложения автоторговли.

Запуск::
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Жизненный цикл:
    startup  → init_db + закрытие позиций, оставшихся от аварийного выхода
               + запуск планировщика
    shutdown → остановка планировщика + закрытие всех открытых позиций
               + dispose_db

Время во всех логах, signals.csv и БД — UTC (core/timeutil.py).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import settings
from core.logging_service import setup_logging
from db.database import dispose_db, init_db

setup_logging()
logger = logging.getLogger("trading")

# Пути к шаблонам/статике (используются роутами позже).
TEMPLATES = Jinja2Templates(directory="app/templates")


def _pricefmt(value) -> str:
    """Адаптивное форматирование цены: больше знаков для дешёвых монет.

    Устраняет баг, когда entry/stop/target дешёвой монеты (0.0781 / 0.0794 / 0.0759)
    округляются до 0.08 и становятся неразличимы на сайте.

    > 1000   → 2 знака    (64000.00)
    > 1      → 4 знака    (1.6800)
    > 0.1    → 5 знаков   (0.07813)
    > 0.001  → 6 знаков   (0.000034)
    иначе    → 8 знаков
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v >= 1000:
        return f"{v:.2f}"
    if v >= 1:
        return f"{v:.4f}"
    if v >= 0.1:
        return f"{v:.5f}"
    if v >= 0.001:
        return f"{v:.6f}"
    return f"{v:.8f}"


# Регистрируем фильтр для всех шаблонов.
TEMPLATES.env.filters["pricefmt"] = _pricefmt


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Полный жизненный цикл приложения (план, этап 8)."""
    setup_logging()
    logger.info("Запуск приложения (env=%s). Время в логах, CSV и БД — UTC",
                settings.app_env)
    await init_db()
    logger.info("БД инициализирована: %s", settings.database_url)

    # Запуск TradingLoop (планировщик APScheduler, этап 8).
    from core.trading_loop import get_loop
    loop = get_loop()

    # Позиции не переживают выход. Если прошлый запуск завершился аварийно
    # (закрыли окно консоли, уснул/выключился компьютер) — shutdown-обработчик
    # не отработал и позиции остались в БД открытыми. Закрываем их сейчас,
    # до старта планировщика.
    try:
        await loop.close_open_positions(
            "старт после аварийной остановки (позиции остались с прошлого запуска)"
        )
    except Exception as exc:
        logger.exception("Не удалось закрыть позиции прошлого запуска: %s", exc)

    await loop.start()

    # Предзагрузка Kronos в фоне (ленивая загрузка занимает ~10 сек,
    # без этого predict() никогда не вызовется из-за проверки is_loaded).
    import asyncio
    from core.prediction_service import get_prediction_service

    async def _warmup_kronos() -> None:
        try:
            pred_svc = get_prediction_service()
            await asyncio.to_thread(lambda: pred_svc.load())
            logger.info("Kronos предзагружен: %s", pred_svc.is_loaded)
            # Автоматический первичный расчёт прогнозов для всех монет сразу после загрузки весов
            asyncio.create_task(loop.refresh_all_predictions())
        except Exception as e:
            logger.warning("Kronos предзагрузка не удалась (CPU fallback?): %s", e)

    asyncio.create_task(_warmup_kronos())

    try:
        yield
    finally:
        logger.info("Остановка приложения")
        # stop() останавливает планировщик и закрывает все открытые позиции.
        await loop.stop()
        try:
            from core.market_service import get_market
            await get_market().close()
        except Exception as exc:
            logger.debug("MarketService.close: %s", exc)
        await dispose_db()
        logger.info("Приложение остановлено")


app = FastAPI(
    title="Kronos Auto-Trading",
    description="Прототип автоторговли криптовалютой на базе Kronos (тестовый режим).",
    version="0.1.0",
    lifespan=lifespan,
)

# Авторизация: middleware проверяет cookie для всех запросов.
# Если пароль не задан (APP_PASSWORD пуст) — авторизация отключена.
from app.auth.middleware import AuthMiddleware  # noqa: E402

app.add_middleware(AuthMiddleware)

# Статика (CSS/JS для HTMX/Plotly).
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ─── Health-check (минимальный эндпоинт для проверки) ────────


@app.get("/api/health")
async def health() -> dict:
    """Health-check с диагностикой планировщика (мониторинг туннеля/простоя).

    Возвращает:
      - status: базовый статус.
      - scheduler_running: работает ли APScheduler (должен быть True).
      - auto_pairs: список пар с включённым авто-режимом (для проверки что
        состояние не потерялось после простоя/рестарта).
      - uptime_seconds: сколько секунд работает процесс.
      - jobs: запланированные cron-задачи с next_run — для отладки пропусков.
    """
    from core.trading_loop import get_loop
    try:
        loop_health = get_loop().get_health()
    except Exception as exc:
        loop_health = {"error": str(exc)}
    return {"status": "ok", "env": settings.app_env, "loop": loop_health}


@app.get("/api/config")
async def get_config() -> dict:
    """Сводка ключевых параметров стратегии (без секретов)."""
    return {
        "exchange": settings.exchange,
        "paper_initial_balance": settings.paper_initial_balance,
        "round_trip_cost": settings.round_trip_cost,
        "min_expected_move": settings.min_expected_move,
        "required_history": settings.required_history,
        "retention_size": settings.retention_size,
        "coins": settings.default_coins,
        "timeframes": settings.default_timeframes,
    }


# ─── Favicon ─────────────────────────────────────────────────

@app.get("/favicon.ico")
async def favicon():
    """Отдаём SVG-фавиконку по запросу /favicon.ico (браузеры и localtunnel
    запрашивают именно .ico, иначе в логах бесконечные 404)."""
    from fastapi.responses import FileResponse
    return FileResponse(
        "app/static/favicon.svg",
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ─── Роуты UI (Jinja2 + HTMX, этап 9) ─────────────────────────

from app.routes import auth, ui  # noqa: E402

app.include_router(auth.router)
app.include_router(ui.router)
