"""Централизованная система логирования для автономной торговли (Kronos).

Обеспечивает сохранение данных для аудита за 2-3+ дня:
Время во всех трёх файлах — UTC, формат YYYY-MM-DD HH:MM:SS (core/timeutil.py).

1. logs/trading.log   — полный ротируемый лог работы системы (FastAPI, APScheduler, DataLifecycle, Kronos).
2. logs/trades.log    — специализированный лог сделок (OPEN, CLOSE, REJECT) для быстрого аудита эффективности.
3. logs/signals.csv   — табличный журнал всех тиков, сигналов, фильтров и решений для статистического анализа.
"""

from __future__ import annotations

import csv
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
import time
from typing import Any

from core.timeutil import TIME_FORMAT

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRADING_LOG_PATH = LOG_DIR / "trading.log"
TRADES_LOG_PATH = LOG_DIR / "trades.log"
SIGNALS_CSV_PATH = LOG_DIR / "signals.csv"

_trades_logger = logging.getLogger("trades")
_csv_lock = threading.Lock()

SIGNALS_CSV_COLUMNS = [
    "timestamp",
    "symbol",
    "timeframe",
    "close",
    "direction",
    "pred_return",
    "pred_slope",
    "s_entry",
    "s_entry_short",
    "s_exit",
    "f_trend",
    "f_kronos",
    "f_momentum",
    "f_rsi",
    "f_pullback",
    "f_news",
    "adx",
    "rsi",
    "action",
    "detail",
    "reject_reason",
]


def setup_logging() -> None:
    """Инициализировать файловые ротируемые обработчики логов."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Основной логгер (trading + root)
    main_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt=TIME_FORMAT,
    )
    # Единое время проекта — UTC (а не локальное время машины).
    main_formatter.converter = time.gmtime

    # Консоль
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(main_formatter)
        root_logger.addHandler(console_handler)

    # Файл trading.log (20 МБ, 5 ротаций = до 100 МБ)
    trading_file_handler = RotatingFileHandler(
        TRADING_LOG_PATH,
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    trading_file_handler.setLevel(logging.INFO)
    trading_file_handler.setFormatter(main_formatter)

    root_logger.setLevel(logging.INFO)
    # При обрыве сети aiohttp/asyncio дублируют каждую DNS-ошибку полным
    # traceback'ом — их отсекаем (обрыв пишет одной строкой NetworkGuard).
    from core.network_guard import DropNetworkNoise
    asyncio_logger = logging.getLogger("asyncio")
    if not any(isinstance(f, DropNetworkNoise) for f in asyncio_logger.filters):
        asyncio_logger.addFilter(DropNetworkNoise())
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', '').endswith("trading.log") for h in root_logger.handlers):
        root_logger.addHandler(trading_file_handler)

    # 2. Логгер сделок trades.log (10 МБ, 5 ротаций)
    trades_formatter = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt=TIME_FORMAT,
    )
    trades_formatter.converter = time.gmtime
    trades_file_handler = RotatingFileHandler(
        TRADES_LOG_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    trades_file_handler.setLevel(logging.INFO)
    trades_file_handler.setFormatter(trades_formatter)

    _trades_logger.setLevel(logging.INFO)
    _trades_logger.propagate = False  # Не спамить в общий trading.log
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', '').endswith("trades.log") for h in _trades_logger.handlers):
        _trades_logger.addHandler(trades_file_handler)

    # 3. Инициализация signals.csv заголовком (если файл новый)
    _init_signals_csv()


def _init_signals_csv() -> None:
    """Создаёт signals.csv с заголовком, если файла ещё нет."""
    with _csv_lock:
        if not SIGNALS_CSV_PATH.exists() or SIGNALS_CSV_PATH.stat().st_size == 0:
            with open(SIGNALS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(SIGNALS_CSV_COLUMNS)


def log_trade_open(
    symbol: str,
    tf: str,
    side: str,
    price: float,
    sl: float,
    tp: float,
    qty: float,
    fee: float,
    cash_after: float,
    reason: str = "entry_signal",
) -> None:
    """Записать открытие позиции в журнал сделок."""
    _trades_logger.info(
        "OPEN   | %-10s | %-3s | %-5s | entry=%-11.4f | sl=%-11.4f | tp=%-11.4f | qty=%-10.6f | fee=$%-6.4f | cash=$%-9.2f | reason=%s",
        symbol, tf, side.upper(), price, sl, tp, qty, fee, cash_after, reason,
    )


def log_trade_close(
    symbol: str,
    tf: str,
    side: str,
    exit_reason: str,
    exit_price: float,
    net_pnl: float,
    pnl_pct: float,
    hold_bars: int,
    fee: float,
    cash_after: float,
) -> None:
    """Записать закрытие позиции в журнал сделок."""
    pnl_sign = "+" if net_pnl >= 0 else ""
    _trades_logger.info(
        "CLOSE  | %-10s | %-3s | %-5s | exit=%-12.4f | pnl=%s$%-8.4f (%s%.2f%%) | hold=%-3d bars | fee=$%-6.4f | cash=$%-9.2f | reason=%s",
        symbol, tf, side.upper(), exit_price, pnl_sign, net_pnl, pnl_sign, pnl_pct * 100, hold_bars, fee, cash_after, exit_reason,
    )


def log_shutdown_summary(context: str, closed: int, total_pnl: float, cash_after: float) -> None:
    """Итог принудительного закрытия позиций при остановке/старте приложения."""
    pnl_sign = "+" if total_pnl >= 0 else ""
    _trades_logger.info(
        "FLAT   | закрыто позиций: %d | pnl=%s$%.4f | cash=$%.2f | %s",
        closed, pnl_sign, total_pnl, cash_after, context,
    )


def log_trade_reject(
    symbol: str,
    tf: str,
    side: str,
    score: float,
    reason: str,
) -> None:
    """Записать причину отказа от сделки в журнал сделок."""
    _trades_logger.info(
        "REJECT | %-10s | %-3s | %-5s | score=%-6.3f | reason=%s",
        symbol, tf, side.upper(), score, reason,
    )


def record_signal_csv(row: dict[str, Any]) -> None:
    """Записать строку в signals.csv для статистического анализа."""
    _init_signals_csv()
    with _csv_lock:
        try:
            with open(SIGNALS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SIGNALS_CSV_COLUMNS)
                # Извлекаем только известные поля
                csv_row = {}
                for col in SIGNALS_CSV_COLUMNS:
                    val = row.get(col)
                    if isinstance(val, float):
                        csv_row[col] = f"{val:.5f}"
                    elif val is None:
                        csv_row[col] = ""
                    else:
                        csv_row[col] = str(val)
                writer.writerow(csv_row)
        except Exception as e:
            logging.getLogger("trading").warning("Не удалось записать в signals.csv: %s", e)
