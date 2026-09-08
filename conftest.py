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
