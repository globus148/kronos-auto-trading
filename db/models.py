"""ORM-модели. Соответствуют схеме данных из плана (раздел 7).

Таблицы:
    coin              — торгуемые монеты
    candle            — OHLCV с retention (раздел 6)
    session           — сессия торговли (тестовая/реальная)
    position          — открытая позиция
    trade             — исполненная сторона сделки (вход/выход)
    equity_point      — точка equity-кривой
    prediction        — кэш прогнозов Kronos (TTL)
    indicator_snapshot— значения индикаторов + скор S_entry/S_exit
    strategy_config   — настройки стратегии (key/value)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    BigInteger,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Базовый класс всех моделей."""


# ─── Перечисления ─────────────────────────────────────────────


class SessionMode(str, Enum):
    paper = "paper"   # тестовый режим
    live = "live"     # реальный (не реализуется в прототипе)


class PositionSide(str, Enum):
    long = "long"
    short = "short"


class PositionStatus(str, Enum):
    open = "open"
    closed = "closed"


class TradeSide(str, Enum):
    buy = "buy"
    sell = "sell"


class TradeReason(str, Enum):
    """Причина исполнения сделки. Для выхода — какое условие сработало (план 4.4)."""
    entry_signal = "entry_signal"
    manual = "manual"
    take_profit = "take_profit"
    trailing_stop = "trailing_stop"
    stop_loss = "stop_loss"
    signal_reversal = "signal_reversal"
    time_stop = "time_stop"


# ─── Монеты ───────────────────────────────────────────────────


class Coin(Base):
    __tablename__ = "coin"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)  # "BTC/USDT"
    enabled: Mapped[bool] = mapped_column(default=True)
    default_tf: Mapped[str] = mapped_column(String(5), default="4h")

    candles: Mapped[list[Candle]] = relationship(back_populates="coin")


# ─── Свечи (retention-таблица, план раздел 6) ─────────────────


class Candle(Base):
    __tablename__ = "candle"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey("coin.id"), index=True)
    tf: Mapped[str] = mapped_column(String(5), index=True)          # "1h" | "4h"
    timestamp: Mapped[int] = mapped_column(BigInteger, index=True)  # ms epoch (открытие свечи)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    coin: Mapped[Coin] = relationship(back_populates="candles")

    __table_args__ = (
        # Уникальность по паре+ТФ+время + быстрый «последние N» (план 6.6).
        Index("uq_candle_coin_tf_ts", "coin_id", "tf", "timestamp", unique=True),
        Index("ix_candle_coin_tf_ts_desc", "coin_id", "tf", "timestamp"),
    )


# ─── Сессия торговли ──────────────────────────────────────────


class Session(Base):
    __tablename__ = "session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[SessionMode] = mapped_column(default=SessionMode.paper)
    initial_balance: Mapped[float] = mapped_column(Float, default=100.0)
    cash: Mapped[float] = mapped_column(Float, default=100.0)        # свободный USDT
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(default=True)

    positions: Mapped[list[Position]] = relationship(back_populates="session")
    equity_points: Mapped[list[EquityPoint]] = relationship(back_populates="session")


# ─── Позиция ──────────────────────────────────────────────────


class Position(Base):
    __tablename__ = "position"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"), index=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey("coin.id"), index=True)
    tf: Mapped[str] = mapped_column(String(5), default="4h")
    side: Mapped[PositionSide] = mapped_column(default=PositionSide.long)
    qty: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    entry_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    stop_price: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    # Текущий trailing-stop (обновляется по мере роста цены, план 4.4.2).
    trailing_stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Сколько свечей позиция уже открыта (для time-stop, план 4.4.5).
    bars_held: Mapped[int] = mapped_column(Integer, default=0)
    # Уровень риска в $ (1% equity) — для расчёта R-multiple выхода.
    risk_amount: Mapped[float] = mapped_column(Float)
    status: Mapped[PositionStatus] = mapped_column(
        default=PositionStatus.open, index=True
    )

    session: Mapped[Session] = relationship(back_populates="positions")
    trades: Mapped[list[Trade]] = relationship(
        back_populates="position", order_by="Trade.executed_at"
    )


# ─── Сделка (каждая сторона = запись) ─────────────────────────


class Trade(Base):
    __tablename__ = "trade"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int] = mapped_column(ForeignKey("position.id"), index=True)
    side: Mapped[TradeSide] = mapped_column()
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, default=0.0)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[TradeReason] = mapped_column(default=TradeReason.manual)
    executed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    position: Mapped[Position] = relationship(back_populates="trades")


# ─── Equity-кривая ────────────────────────────────────────────


class EquityPoint(Base):
    __tablename__ = "equity_point"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    equity: Mapped[float] = mapped_column(Float)            # cash + позиции по рынку
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)

    session: Mapped[Session] = relationship(back_populates="equity_points")


# ─── Кэш прогнозов Kronos (TTL) ───────────────────────────────


class Prediction(Base):
    __tablename__ = "prediction"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey("coin.id"), index=True)
    tf: Mapped[str] = mapped_column(String(5), index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    horizon: Mapped[int] = mapped_column(Integer)            # N свечей вперёд
    payload_json: Mapped[str] = mapped_column(Text)          # OHLCV прогноза
    actual_return: Mapped[float | None] = mapped_column(Float, nullable=True)  # для точности


# ─── Снапшот индикаторов + скор ───────────────────────────────


class IndicatorSnapshot(Base):
    __tablename__ = "indicator_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey("coin.id"), index=True)
    tf: Mapped[str] = mapped_column(String(5), index=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, index=True)  # ms epoch
    rsi: Mapped[float] = mapped_column(Float)
    macd: Mapped[float] = mapped_column(Float)
    macd_signal: Mapped[float] = mapped_column(Float)
    macd_hist: Mapped[float] = mapped_column(Float)
    ema20: Mapped[float] = mapped_column(Float)
    ema50: Mapped[float] = mapped_column(Float)
    atr: Mapped[float] = mapped_column(Float)
    adx: Mapped[float] = mapped_column(Float)
    bb_upper: Mapped[float] = mapped_column(Float)
    bb_lower: Mapped[float] = mapped_column(Float)
    s_entry: Mapped[float] = mapped_column(Float, default=0.0)
    s_exit: Mapped[float] = mapped_column(Float, default=0.0)


# ─── Настройки стратегии ──────────────────────────────────────


class StrategyConfig(Base):
    __tablename__ = "strategy_config"

    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text)


# ─── Кошелёк биржи (live-режим, реальные деньги) ──────────────


class Wallet(Base):
    """Привязка API-ключей биржи для live-торговли реальными деньгами.

    api_secret хранится в зашифрованном виде (Fernet, core/wallet_service.py).
    Один кошелёк может быть `is_default` — именно он используется live-режимом.
    """

    __tablename__ = "wallet"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(100))             # "Binance main"
    exchange: Mapped[str] = mapped_column(String(30), default="binance")
    api_key: Mapped[str] = mapped_column(String(200))
    # Зашифрованный Fernet-токен api_secret (не хранится в открытом виде).
    api_secret_enc: Mapped[str] = mapped_column(Text)
    is_default: Mapped[bool] = mapped_column(default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


# ─── Авто-торговля: персистентное состояние (M) ───────────────


class AutoConfig(Base):
    """Включён ли авто-режим для пары/ТФ (переживает рестарт процесса).

    Раньше это был in-memory dict в TradingLoop — при любом перезапуске
    сервера все авто-переключатели сбрасывались в OFF. Теперь сохраняем
    в БД: 'pair' = "BTC/USDT:4h", 'enabled' = bool.
    """

    __tablename__ = "auto_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pair: Mapped[str] = mapped_column(String(30), unique=True, index=True)  # "BTC/USDT:4h"
    enabled: Mapped[bool] = mapped_column(default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
