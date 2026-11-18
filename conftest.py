"""Конфигурация pytest для всего проекта.

ВАЖНО: этот файл загружается ДО импорта любых тестов и модулей проекта.
Здесь мы переключаем тесты на отдельную in-memory SQLite БД, чтобы они
НЕ загрязняли рабочую ./data/trading.db (где живут реальные монеты/свечи
и куда тесты раньше случайно писали TEST*/USDT).

DATABASE_URL выставляется через os.environ до того, как config.settings
считает .env — тогда lru_cache зафиксирует тестовый URL.
"""
import os

# Тестовая БД: in-memory SQLite. Ставим ДО импорта config/db.database.
# aiosqlite поддерживает ":memory:" через special URL.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"

import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _isolate_logs(tmp_path_factory):
    """Тесты не пишут в рабочую папку logs/ (signals.csv, trades.log).

    Раньше run_once() в тестах дописывал тестовые строки в боевой
    logs/signals.csv. Перенаправляем пути логов во временную папку.
    """
    import core.logging_service as ls

    d = tmp_path_factory.mktemp("logs")
    ls.LOG_DIR = d
    ls.TRADING_LOG_PATH = d / "trading.log"
    ls.TRADES_LOG_PATH = d / "trades.log"
    ls.SIGNALS_CSV_PATH = d / "signals.csv"
    yield

@pytest.fixture(autouse=True)
def _fresh_network_guard():
    """Каждый тест начинает с «связь есть» (guard — глобальный синглтон)."""
    import core.network_guard as ng

    ng._guard = None
    yield
    ng._guard = None

