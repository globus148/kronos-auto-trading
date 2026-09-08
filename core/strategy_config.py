"""Редактируемые настройки стратегии (живые пороги).

Позволяет менять ключевые пороги стратегии с сайта (страница /settings) без
перезапуска приложения. Значения хранятся в таблице `strategy_config`
(key → value_json), кешируются в памяти на TTL секунд (чтобы не ходить в БД на
каждый тик/каждый суб-сигнал), и имеют фоллбэк на `config.settings`.

Архитектура:
    - get_threshold(key)          → float: из кеша/БД или дефолт config.settings.
    - get_thresholds()            → dict: все пороги сразу (для UI и snapshot).
    - save_thresholds(updates)    → None: перезаписать значения в БД + сброс кеша.

Пороги, которые можно редактировать (см. THRESHOLD_KEYS):
    entry_threshold, strategy_min_rr, strategy_min_adx,
    risk_fraction, max_position_pct, atr_k_stop, atr_k_take,
    max_hold_periods, commission_rate, slippage_rate, margin_of_safety.

Остальные настройки (модель Kronos, монеты, БД) остаются в config.settings —
их менять на лету нельзя.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.database import get_session
from db.models import StrategyConfig

logger = logging.getLogger("trading.config")

# ─── Описание редактируемых порогов ──────────────────────────

# key → (тип, дефолт из config.settings, min, max, подпись для UI).
THRESHOLD_KEYS: dict[str, dict[str, Any]] = {
    "entry_threshold": {
        "label": "Порог S_entry (вход)", "type": "float",
        "min": 0.0, "max": 1.0, "step": 0.01,
        "hint": "Композитный сигнал входа ≥ этого значения. Ниже → чаще входит (0.5–0.8).",
    },
    "strategy_min_rr": {
        "label": "Мин. Risk/Reward", "type": "float",
        "min": 0.5, "max": 5.0, "step": 0.1,
        "hint": "Минимальное отношение прибыль/риск. Ниже → больше сделок, но рискованнее.",
    },
    "strategy_min_adx": {
        "label": "Мин. ADX (сила тренда)", "type": "float",
        "min": 0.0, "max": 60.0, "step": 1.0,
        "hint": "ADX показывает силу тренда. Ниже → вход и во флэте (0–40).",
    },
    "risk_fraction": {
        "label": "Риск на сделку (% equity)", "type": "percent",
        "min": 0.1, "max": 10.0, "step": 0.1,
        "hint": "Доля капитала на одну сделку. 1% = консервативно, 2% = умеренно.",
    },
    "max_position_pct": {
        "label": "Макс. позиция (% equity)", "type": "percent",
        "min": 1.0, "max": 100.0, "step": 1.0,
        "hint": "Максимум капитала в одной позиции.",
    },
    "atr_k_stop": {
        "label": "ATR Stop k", "type": "float",
        "min": 0.5, "max": 4.0, "step": 0.1,
        "hint": "Множитель ATR для стоп-лосса (1.0–2.0).",
    },
    "atr_k_take": {
        "label": "ATR Take k", "type": "float",
        "min": 1.0, "max": 8.0, "step": 0.1,
        "hint": "Множитель ATR для тейк-профита (2.0–4.0).",
    },
    "max_hold_periods": {
        "label": "Макс. свечей в позиции", "type": "int",
        "min": 1, "max": 500, "step": 1,
        "hint": "Принудительный выход (time-stop) после N свечей.",
    },
    "margin_of_safety": {
        "label": "Запас безубыточности", "type": "float",
        "min": 0.0, "max": 0.05, "step": 0.001,
        "hint": "Компонент min_expected_move (round_trip + запас).",
    },
}


def _default_for(key: str) -> float:
    """Дефолтное значение порога из config.settings."""
    return float(getattr(settings, key))


def _is_percent(key: str) -> bool:
    """Поле хранится как дробь 0–1, в UI показывается как процент 0–100."""
    return THRESHOLD_KEYS.get(key, {}).get("type") == "percent"


def _to_display(key: str, value: float) -> float:
    """Внутреннее значение (дробь) → значение для UI (процент для percent-полей)."""
    return value * 100.0 if _is_percent(key) else value


def _from_display(key: str, value: float) -> float:
    """Значение из UI (процент) → внутреннее (дробь) для percent-полей."""
    return value / 100.0 if _is_percent(key) else value


# ─── In-memory кеш с TTL ─────────────────────────────────────

_cache: dict[str, tuple[float, float]] = {}  # key → (value, cached_at)
_cache_ttl = 5.0  # секунд
_cache_lock = asyncio.Lock()


async def get_threshold(key: str) -> float:
    """Получить порог по ключу (из кеша/БД/дефолта)."""
    now = time.time()
    cached = _cache.get(key)
    if cached is not None and (now - cached[1]) < _cache_ttl:
        return cached[0]

    value = await _load_from_db(key)
    _cache[key] = (value, now)
    return value


async def _load_from_db(key: str) -> float:
    """Загрузить значение из БД или вернуть дефолт."""
    try:
        async with get_session() as db:
            row = (await db.execute(
                select(StrategyConfig).where(StrategyConfig.key == key)
            )).scalar_one_or_none()
            if row is not None:
                return float(json.loads(row.value_json))
    except Exception as e:
        logger.debug("get_threshold(%s): фоллбэк на дефолт (%s)", key, e)
    return _default_for(key)


async def get_thresholds() -> dict[str, float]:
    """Все редактируемые пороги сразу (для snapshot — ВНУТРЕННИЙ формат, дроби).

    Snapshot отдают дроби (0.01 для 1%), т.к. StrategyEngine/RiskManager
    ждут именно дроби. UI-слой использует get_thresholds_for_ui().
    Делает один проход, без N запросов к БД.
    """
    # Сначала пробуем загрузить все записи из БД одним запросом.
    db_values: dict[str, float] = {}
    try:
        async with get_session() as db:
            rows = (await db.execute(
                select(StrategyConfig).where(
                    StrategyConfig.key.in_(list(THRESHOLD_KEYS))
                )
            )).scalars().all()
            for r in rows:
                db_values[r.key] = float(json.loads(r.value_json))
    except Exception as e:
        logger.debug("get_thresholds: фоллбэк на дефолты (%s)", e)

    result: dict[str, float] = {}
    now = time.time()
    for key in THRESHOLD_KEYS:
        if key in db_values:
            result[key] = db_values[key]
            _cache[key] = (db_values[key], now)
        else:
            result[key] = _default_for(key)
            _cache[key] = (result[key], now)
    return result


async def get_thresholds_for_ui() -> dict[str, float]:
    """Пороги для веб-формы: percent-поля конвертированы в проценты (1.0 = 1%).

    Внутренне всегда хранятся дроби (0.01), но в форме настроек человек видит
    и вводит проценты (1.0). Эта функция применяет конвертацию для отображения.
    """
    raw = await get_thresholds()
    return {key: _to_display(key, val) for key, val in raw.items()}


async def save_thresholds(
    updates: dict[str, float],
    form_is_percent: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    """Сохранить пороги в БД (upsert) и сбросить кеш.

    Args:
        updates: значения из формы. Если form_is_percent=True (по умолчанию),
            percent-поля (risk_fraction, max_position_pct) приходят как проценты
            (1.0 = 1%) и конвертируются в дробь (0.01) перед сохранением.
        form_is_percent: True для значений из веб-формы, False для внутренних.

    Возвращает кортеж (saved_display, clamped):
        - saved_display: итоговые значения в формате UI (проценты для percent-полей).
        - clamped: {key: (введено, сохранено)} для полей, зажатых в [min, max].
    """
    cleaned: dict[str, float] = {}       # внутренний формат (дроби) для БД
    entered_display: dict[str, float] = {}  # что ввёл юзер (формат UI)
    clamped: dict[str, tuple[float, float]] = {}

    for key, raw in updates.items():
        if key not in THRESHOLD_KEYS:
            continue
        meta = THRESHOLD_KEYS[key]
        try:
            entered = float(raw)
        except (TypeError, ValueError):
            continue
        entered_display[key] = entered
        # Клампинг в формате UI (min/max в THRESHOLD_KEYS заданы как проценты
        # для percent-полей — это то, что видит юзер в форме).
        clamped_val = max(meta["min"], min(meta["max"], entered))
        if clamped_val != entered:
            clamped[key] = (entered, clamped_val)
        # Конвертация в внутренний формат (дробь) для БД.
        internal = _from_display(key, clamped_val) if form_is_percent else clamped_val
        cleaned[key] = internal

    if not cleaned:
        return await get_thresholds_for_ui(), clamped

    async with get_session() as db:
        for key, val in cleaned.items():
            # SQLite upsert.
            stmt = sqlite_insert(StrategyConfig).values(
                key=key, value_json=json.dumps(val),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[StrategyConfig.key],
                set_={"value_json": stmt.excluded.value_json},
            )
            await db.execute(stmt)
        await db.commit()

    # Сбрасываем кеш изменённых ключей.
    async with _cache_lock:
        for key in cleaned:
            _cache.pop(key, None)

    logger.info("Сохранены пороги стратегии (внутр. формат): %s", cleaned)
    saved_display = await get_thresholds_for_ui()
    return saved_display, clamped


def invalidate_cache() -> None:
    """Сбросить весь кеш порогов (после ручного изменения БД)."""
    _cache.clear()


# ─── Синглтон-помощник для синхронных контекстов ─────────────
# StrategyEngine/RiskManager создаются синхронно и читают пороги в sync-методах.
# Поэтому даём им снапшот порогов, который обновляется раз в TTL.

_snapshot: dict[str, float] | None = None
_snapshot_at: float = 0.0


async def refresh_snapshot() -> dict[str, float]:
    """Обновить синхронный снапшот порогов (вызывать из async-кода).
    StrategyEngine.get_threshold(key) читает из снапшота без await.
    """
    global _snapshot, _snapshot_at
    _snapshot = await get_thresholds()
    _snapshot_at = time.time()
    return _snapshot


def get_snapshot() -> dict[str, float]:
    """Синхронное чтение снапшота. Если устарел — вернёт дефолты.
    trading_loop.run_once вызывает refresh_snapshot() в начале каждого тика.
    """
    if _snapshot is None:
        # Первый запуск или кеш сброшен — отдаём дефолты (без await).
        return {k: _default_for(k) for k in THRESHOLD_KEYS}
    # Если снапшот протух — тоже отдаём, обновится на следующем тике.
    return _snapshot
