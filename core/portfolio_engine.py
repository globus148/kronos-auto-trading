"""PortfolioEngine — журнал сделок, PnL, комиссии, equity-кривая (план 4.7, 6).

Отвечает за:
    - Создание/закрытие позиций с учётом комиссий и slippage (план 4.1)
    - Ведение cash-баланса (spot long + кэш-позиции)
    - Расчёт PnL закрытых и нереализованного PnL открытых
    - Запись equity_point на каждом тике (для equity-кривой в UI)
    - Сводные метрики аналитики (win rate, Sharpe, drawdown и т.д.)

Тестовый режим: старт с $100, никаких реальных денег.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.models import (
    Coin,
    EquityPoint,
    Position,
    PositionSide,
    PositionStatus,
    Session,
    SessionMode,
    Trade,
    TradeReason,
    TradeSide,
)
from core.risk_manager import PositionSizing

logger = logging.getLogger("trading.portfolio")

# Минимальный интервал между пирамидингом (добавлением позиции)
# по одной и той же монете в одну сторону. Без cooldown polling/live
# открывает 3+ позиций за секунду — бессмысленное утроение риска.
PYRAMIDING_COOLDOWN_SEC = 300  # 5 минут

# Минимальный интервал между закрытием позиции и новым входом по той же
# монете. Без cooldown стратегия открывает новую позицию сразу после
# time_stop/stop_loss — churn на комиссиях без выгоды (сигнал обычно всё
# ещё тот же, что привёл к убытку).
POST_EXIT_COOLDOWN_SEC = 3600  # 1 час


# ─── Результат сделки ────────────────────────────────────────

@dataclass
class TradeExecution:
    """Результат исполнения одной стороны сделки."""
    trade: Trade
    fee: float
    slippage_cost: float
    net_value: float        # фактическая стоимость с учётом издержек


@dataclass
class ClosedPositionResult:
    """Итог закрытой позиции."""
    position: Position
    entry_trade: Trade
    exit_trade: Trade
    gross_pnl: float        # PnL без комиссий
    total_fees: float       # суммарные комиссии (вход+выход)
    total_slippage: float
    net_pnl: float          # чистый PnL
    pnl_pct: float          # % от вложенного
    hold_bars: int
    exit_reason: str


@dataclass
class AnalyticsReport:
    """Сводные метрики аналитики (план 4.7)."""
    total_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    expectancy: float
    max_drawdown: float
    sharpe: float
    sortino: float
    total_fees: float
    fee_drag_pct: float          # какой % PnL съедает комиссия


class PortfolioEngine:
    """Управление позициями, сделками и equity (paper trading)."""

    def __init__(
        self,
        commission_rate: float | None = None,
        slippage_rate: float | None = None,
    ) -> None:
        self.commission_rate = commission_rate or settings.commission_rate
        self.slippage_rate = slippage_rate or settings.slippage_rate

    # ─── Cooldown-проверки ────────────────────────────────────

    async def _in_post_exit_cooldown(
        self, db: AsyncSession, session: Session, coin: Coin,
    ) -> bool:
        """Проверить, был ли недавний выход по этой монете.

        Возвращает True, если последняя закрытая позиция по монете
        закрыта менее POST_EXIT_COOLDOWN_SEC назад — новый вход блокируется
        (churn-защита).
        """
        last_closed = await self.get_last_closed_position_by_coin(db, session, coin)
        if last_closed is None:
            return False
        # Берём время последнего exit-trade (explicit query, не lazy-loading).
        stmt = (
            select(func.max(Trade.executed_at))
            .where(Trade.position_id == last_closed.id, Trade.pnl != 0.0)
        )
        last_exit_time = (await db.execute(stmt)).scalar()
        if last_exit_time is None:
            # Нет exit-трейда — используем entry_at как upper bound.
            ref_time = last_closed.entry_at
        else:
            ref_time = last_exit_time
        now = datetime.now(timezone.utc)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)
        age_sec = (now - ref_time).total_seconds()
        if age_sec < POST_EXIT_COOLDOWN_SEC:
            logger.info(
                "По %s post-exit cooldown (%.0fс < %dс) — пропускаем вход",
                coin.symbol, age_sec, POST_EXIT_COOLDOWN_SEC,
            )
            return True
        return False

    # ─── Создание/получение сессии ───────────────────────────

    async def get_or_create_session(
        self, db: AsyncSession, mode: SessionMode = SessionMode.paper,
    ) -> Session:
        """Получить активную сессию или создать новую с $100 стартового баланса."""
        stmt = (
            select(Session)
            .where(Session.is_active == True, Session.mode == mode)
            .order_by(Session.created_at.desc())
        )
        result = await db.execute(stmt)
        session = result.scalars().first()
        if session is None:
            # В live-режиме стартовый cash = 0 (реальный баланс подгрузится с
            # биржи при первой возможности). В paper — settings.paper_initial_balance.
            initial = 0.0 if mode == SessionMode.live else settings.paper_initial_balance
            session = Session(
                mode=mode,
                initial_balance=initial,
                cash=initial,
                is_active=True,
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
            # Записываем стартовую точку equity.
            await self.record_equity(db, session)
            logger.info(
                "Создана сессия #%s mode=%s balance=$%.2f",
                session.id, mode.value, session.initial_balance,
            )
        return session

    # ─── Открытие позиции ────────────────────────────────────

    async def open_position(
        self,
        db: AsyncSession,
        session: Session,
        coin: Coin,
        sizing: PositionSizing,
        timeframe: str,
        reason: TradeReason = TradeReason.entry_signal,
        force: bool = False,
    ) -> tuple[Position, TradeExecution] | None:
        """Открыть LONG-позицию. Возвращает (position, entry_trade) или None.

        В paper-режиме — симуляция (комиссия + slippage на кэш).
        В live-режиме — реальный рыночный ордер через биржу (LiveBroker).
        """
        # Диспетчер по режиму сессии.
        if session.mode == SessionMode.live:
            return await self._open_position_live(
                db, session, coin, sizing, timeframe, reason,
            )

        if sizing.qty <= 0 or sizing.position_value <= 0:
            logger.warning("open_position: некорректный sizing, пропуск")
            return None

        # Проверка: можно ли добавить ещё одну long-позицию (pyramiding).
        # Нельзя открывать long, если уже есть short по этой монете.
        # Разрешаем до max_positions_per_coin длинных позиций (усреднение/добавление).
        max_positions_per_coin = 3
        existing = await self.get_open_positions_by_coin(db, session, coin)
        same_side = [p for p in existing if p.side != PositionSide.short]
        opposite_side = [p for p in existing if p.side == PositionSide.short]
        if opposite_side:
            logger.info(
                "По %s уже есть SHORT — нельзя открыть long одновременно", coin.symbol,
            )
            return None
        if len(same_side) >= max_positions_per_coin:
            logger.info(
                "По %s уже %d long-позиций (лимит %d) — пропускаем вход",
                coin.symbol, len(same_side), max_positions_per_coin,
            )
            return None
        # Cooldown pyramiding: не открываем новую позицию, если предыдущая
        # по этой монете открыта меньше PYRAMIDING_COOLDOWN секунд назад.
        # Иначе polling/live-обновления открывают 3 позиции за секунду по одной
        # цене — это бессмысленное утроение риска без выгоды.
        # force=True (ручной вход из UI / тесты) — пропускаем cooldown.
        if not force and same_side:
            last_entry = max(p.entry_at for p in same_side if p.entry_at)
            if last_entry is not None:
                now = datetime.now(timezone.utc)
                # entry_at может быть naive (utcnow) — приводим к aware для надёжности.
                if last_entry.tzinfo is None:
                    last_entry = last_entry.replace(tzinfo=timezone.utc)
                age_sec = (now - last_entry).total_seconds()
                if age_sec < PYRAMIDING_COOLDOWN_SEC:
                    logger.info(
                        "По %s pyramiding cooldown (%.0fс < %dс) — пропускаем вход",
                        coin.symbol, age_sec, PYRAMIDING_COOLDOWN_SEC,
                    )
                    return None

        # Post-exit cooldown: не открываем новую long, если позиция по этой
        # монете была закрыта меньше POST_EXIT_COOLDOWN секунд назад — иначе
        # churn (close → re-enter по тому же сигналу) сжигает cash на комиссиях.
        if not force and await self._in_post_exit_cooldown(db, session, coin):
            return None

        # Проверка cash.
        if session.cash < sizing.position_value:
            logger.info(
                "Недостаточно cash ($%.2f < $%.2f) — пропускаем",
                session.cash, sizing.position_value,
            )
            return None

        # Исполнение: цена с учётом slippage (покупаем дороже).
        fill_price = sizing_qty_to_fill_price(sizing, side="buy")
        gross_value = sizing.qty * fill_price
        fee = gross_value * self.commission_rate
        slippage_cost = sizing.qty * (fill_price - sizing.position_value / sizing.qty)
        net_cost = gross_value + fee

        # Списываем cash.
        session.cash -= net_cost

        # Создаём позицию.
        position = Position(
            session_id=session.id, coin_id=coin.id, tf=timeframe,
            side="long",
            qty=sizing.qty, entry_price=fill_price,
            stop_price=sizing.stop_price,
            target_price=sizing.target_price,
            risk_amount=sizing.risk_amount,
            bars_held=0, status=PositionStatus.open,
        )
        db.add(position)
        await db.flush()  # получаем position.id

        # Записываем trade на вход.
        entry_trade = Trade(
            position_id=position.id, side=TradeSide.buy,
            qty=sizing.qty, price=fill_price,
            fee=fee, slippage=slippage_cost,
            pnl=0.0, reason=reason,
        )
        db.add(entry_trade)
        await db.flush()  # чтобы trade был виден в select (autoflush=False)

        execution = TradeExecution(
            trade=entry_trade, fee=fee,
            slippage_cost=slippage_cost, net_value=net_cost,
        )
        logger.info(
            "ОТКРЫТА позиция #%s %s qty=%.6f @ $%.2f fee=$%.4f cash=$%.2f",
            position.id, coin.symbol, sizing.qty, fill_price, fee, session.cash,
        )
        return position, execution

    # ─── Ручное открытие (из модалки UI: сторона + qty + цена) ─

    async def open_manual_position(
        self,
        db: AsyncSession,
        session: Session,
        coin: Coin,
        side: str,
        qty: float,
        entry_price: float,
        timeframe: str,
        atr: float | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
        reason: TradeReason = TradeReason.manual,
    ) -> tuple[Position, TradeExecution] | None:
        """Открыть позицию вручную с явно заданными qty и ценой.

        В отличие от open_position (всё считает RiskManager), здесь трейдер
        сам указывает сторону, количество и цену входа. Стоп/тейк:
          - если переданы stop_price/target_price (из UI) — берём их;
          - иначе считаем от ATR (если передан);
          - иначе нули (только ручное закрытие).

        Args:
            side: "long" | "short".
            qty: количество base (монет).
            entry_price: цена входа (текущая или рыночная).
            atr: значение ATR для расчёта стопов (опционально).
            stop_price: стоп-лосс из UI (переопределяет расчёт по ATR).
            target_price: тейк-профит из UI (переопределяет расчёт по ATR).
        """
        if qty <= 0 or entry_price <= 0:
            logger.warning("open_manual: некорректные qty/price, пропуск")
            return None

        side_norm = (side or "long").lower()
        is_short = side_norm == "short"

        # Live spot не поддерживает шорт.
        if is_short and session.mode == SessionMode.live:
            logger.info("open_manual: live spot не поддерживает шорт, пропуск")
            return None

        position_value = qty * entry_price

        # Проверка cash.
        if session.cash < position_value:
            logger.info(
                "open_manual: недостаточно cash ($%.2f < $%.2f) — пропускаем",
                session.cash, position_value,
            )
            return None

        # Стоп/тейк: приоритет у ручных значений из UI, затем ATR, иначе нули.
        if stop_price is not None and stop_price > 0:
            final_stop = stop_price
        elif atr and atr > 0:
            final_stop = entry_price + settings.atr_k_stop * atr if is_short else entry_price - settings.atr_k_stop * atr
        else:
            final_stop = 0.0
        if target_price is not None and target_price > 0:
            final_target = target_price
        elif atr and atr > 0:
            final_target = entry_price - settings.atr_k_take * atr if is_short else entry_price + settings.atr_k_take * atr
        else:
            final_target = 0.0

        risk_amount = position_value * settings.risk_fraction

        if is_short:
            return await self._open_manual_short(
                db, session, coin, qty, entry_price, timeframe,
                final_stop, final_target, risk_amount, reason,
            )
        return await self._open_manual_long(
            db, session, coin, qty, entry_price, timeframe,
            final_stop, final_target, risk_amount, reason,
        )

    async def _open_manual_long(
        self,
        db: AsyncSession,
        session: Session,
        coin: Coin,
        qty: float,
        entry_price: float,
        timeframe: str,
        stop_price: float,
        target_price: float,
        risk_amount: float,
        reason: TradeReason,
    ) -> tuple[Position, TradeExecution] | None:
        """Ручное открытие LONG (paper-расчёт: без slippage, fee по ставке)."""
        gross_value = qty * entry_price
        fee = gross_value * self.commission_rate
        net_cost = gross_value + fee
        session.cash -= net_cost

        position = Position(
            session_id=session.id, coin_id=coin.id, tf=timeframe,
            side=PositionSide.long,
            qty=qty, entry_price=entry_price,
            stop_price=stop_price, target_price=target_price,
            risk_amount=risk_amount,
            bars_held=0, status=PositionStatus.open,
        )
        db.add(position)
        await db.flush()

        entry_trade = Trade(
            position_id=position.id, side=TradeSide.buy,
            qty=qty, price=entry_price, fee=fee,
            slippage=0.0, pnl=0.0, reason=reason,
        )
        db.add(entry_trade)
        await db.flush()

        execution = TradeExecution(
            trade=entry_trade, fee=fee, slippage_cost=0.0, net_value=net_cost,
        )
        logger.info(
            "РУЧНОЙ LONG #%s %s qty=%.8f @ $%.4f fee=$%.6f cash=$%.2f",
            position.id, coin.symbol, qty, entry_price, fee, session.cash,
        )
        return position, execution

    async def _open_manual_short(
        self,
        db: AsyncSession,
        session: Session,
        coin: Coin,
        qty: float,
        entry_price: float,
        timeframe: str,
        stop_price: float,
        target_price: float,
        risk_amount: float,
        reason: TradeReason,
    ) -> tuple[Position, TradeExecution] | None:
        """Ручное открытие SHORT (paper-обеспечение = position_value)."""
        gross_value = qty * entry_price
        fee = gross_value * self.commission_rate
        # Резервируем обеспечение (возвращается при закрытии ± PnL).
        session.cash -= gross_value

        position = Position(
            session_id=session.id, coin_id=coin.id, tf=timeframe,
            side=PositionSide.short,
            qty=qty, entry_price=entry_price,
            stop_price=stop_price, target_price=target_price,
            risk_amount=risk_amount,
            bars_held=0, status=PositionStatus.open,
        )
        db.add(position)
        await db.flush()

        entry_trade = Trade(
            position_id=position.id, side=TradeSide.sell,
            qty=qty, price=entry_price, fee=fee,
            slippage=0.0, pnl=0.0, reason=reason,
        )
        db.add(entry_trade)
        await db.flush()

        execution = TradeExecution(
            trade=entry_trade, fee=fee, slippage_cost=0.0, net_value=gross_value,
        )
        logger.info(
            "РУЧНОЙ SHORT #%s %s qty=%.8f @ $%.4f fee=$%.6f cash=$%.2f",
            position.id, coin.symbol, qty, entry_price, fee, session.cash,
        )
        return position, execution

    # ─── Открытие SHORT-позиции ───────────────────────────

    async def open_short_position(
        self,
        db: AsyncSession,
        session: Session,
        coin: Coin,
        sizing: PositionSizing,
        timeframe: str,
        reason: TradeReason = TradeReason.entry_signal,
        force: bool = False,
    ) -> tuple[Position, TradeExecution] | None:
        """Открыть SHORT-позицию (paper trading).

        Модель paper-обеспечения: при открытии short «продаём» qty монет,
        которых у нас нет. Резервируем обеспечение = position_value в cash
        (фиктивно списываем), при закрытии возвращаем обеспечение + PnL.

        Вход-сделка side=sell (шорт открывается продажей).

        В live-режиме реальные шорты на spot-бирже невозможны (нет маржи) —
        метод возвращает None (шорт только в paper).
        """
        # Live spot не поддерживает шорт → пропускаем.
        if session.mode == SessionMode.live:
            logger.info("open_short_position: live spot не поддерживает шорт, пропуск")
            return None

        if sizing.qty <= 0 or sizing.position_value <= 0:
            logger.warning("open_short_position: некорректный sizing, пропуск")
            return None

        # Не открываем, если уже есть любая открытая позиция по монете.
        # Нельзя открывать short, если уже есть long по этой монете.
        # Разрешаем до max_positions_per_coin short-позиций.
        max_positions_per_coin = 3
        existing = await self.get_open_positions_by_coin(db, session, coin)
        same_side = [p for p in existing if p.side == PositionSide.short]
        opposite_side = [p for p in existing if p.side != PositionSide.short]
        if opposite_side:
            logger.info(
                "По %s уже есть LONG — нельзя открыть short одновременно", coin.symbol,
            )
            return None
        if len(same_side) >= max_positions_per_coin:
            logger.info(
                "По %s уже %d short-позиций (лимит %d) — пропускаем",
                coin.symbol, len(same_side), max_positions_per_coin,
            )
            return None

        # Cooldown pyramiding: не открываем новую short-позицию, если
        # предыдущая по этой монете открыта меньше PYRAMIDING_COOLDOWN
        # секунд назад.
        # force=True (ручной вход из UI / тесты) — пропускаем cooldown.
        if not force and same_side:
            last_entry = max(p.entry_at for p in same_side if p.entry_at)
            if last_entry is not None:
                now = datetime.now(timezone.utc)
                if last_entry.tzinfo is None:
                    last_entry = last_entry.replace(tzinfo=timezone.utc)
                age_sec = (now - last_entry).total_seconds()
                if age_sec < PYRAMIDING_COOLDOWN_SEC:
                    logger.info(
                        "По %s short pyramiding cooldown (%.0fс < %dс) — пропускаем",
                        coin.symbol, age_sec, PYRAMIDING_COOLDOWN_SEC,
                    )
                    return None

        # Post-exit cooldown: не открываем новый short, если позиция по этой
        # монете была закрыта меньше POST_EXIT_COOLDOWN секунд назад.
        if not force and await self._in_post_exit_cooldown(db, session, coin):
            return None

        # Проверка обеспечения (cash).
        if session.cash < sizing.position_value:
            logger.info(
                "Недостаточно cash для short ($%.2f < $%.2f) — пропускаем",
                session.cash, sizing.position_value,
            )
            return None

        # Исполнение: продаём дешевле из-за slippage (short fill хуже входа).
        fill_price = sizing.position_value / sizing.qty * (1.0 - settings.slippage_rate)
        gross_value = sizing.qty * fill_price
        fee = gross_value * self.commission_rate
        slippage_cost = sizing.qty * (sizing.position_value / sizing.qty - fill_price)
        # Резервируем обеспечение в cash (возвращается при закрытии ± PnL).
        session.cash -= sizing.position_value

        # Создаём short-позицию.
        position = Position(
            session_id=session.id, coin_id=coin.id, tf=timeframe,
            side=PositionSide.short,
            qty=sizing.qty, entry_price=fill_price,
            stop_price=sizing.stop_price,
            target_price=sizing.target_price,
            risk_amount=sizing.risk_amount,
            bars_held=0, status=PositionStatus.open,
        )
        db.add(position)
        await db.flush()  # получаем position.id

        # Trade на вход short = продажа.
        entry_trade = Trade(
            position_id=position.id, side=TradeSide.sell,
            qty=sizing.qty, price=fill_price,
            fee=fee, slippage=slippage_cost,
            pnl=0.0, reason=reason,
        )
        db.add(entry_trade)
        await db.flush()  # чтобы trade был виден в select (autoflush=False)

        execution = TradeExecution(
            trade=entry_trade, fee=fee,
            slippage_cost=slippage_cost, net_value=sizing.position_value,
        )
        logger.info(
            "ОТКРЫТ SHORT #%s %s qty=%.6f @ $%.2f fee=$%.4f cash=$%.2f",
            position.id, coin.symbol, sizing.qty, fill_price, fee, session.cash,
        )
        return position, execution

    # ─── Закрытие позиции ────────────────────────────────────

    async def close_position(
        self,
        db: AsyncSession,
        session: Session,
        position: Position,
        coin: Coin,
        current_price: float,
        reason: TradeReason,
    ) -> ClosedPositionResult | None:
        """Закрыть позицию по рынку (long или short). Считает PnL с комиссиями."""
        if position.status != PositionStatus.open:
            return None

        # Диспетчер по режиму сессии.
        if session.mode == SessionMode.live:
            return await self._close_position_live(
                db, session, position, coin, reason,
            )

        is_short = position.side == PositionSide.short

        entry_trade = await self._get_entry_trade(db, position)
        if entry_trade is None:
            return None

        if is_short:
            return await self._close_short(
                db, session, position, coin, current_price, reason, entry_trade,
            )
        return await self._close_long(
            db, session, position, coin, current_price, reason, entry_trade,
        )

    async def _close_long(
        self,
        db: AsyncSession,
        session: Session,
        position: Position,
        coin: Coin,
        current_price: float,
        reason: TradeReason,
        entry_trade: Trade,
    ) -> ClosedPositionResult | None:
        """Закрыть LONG: продаём дешевле из-за slippage."""
        # Исполнение: продаём дешевле из-за slippage.
        fill_price = current_price * (1.0 - self.slippage_rate)
        gross_value = position.qty * fill_price
        fee = gross_value * self.commission_rate
        net_proceeds = gross_value - fee

        # Добавляем cash.
        session.cash += net_proceeds

        # Закрываем позицию.
        position.status = PositionStatus.closed

        # PnL.
        # entry_cost = qty * entry_fill_price + entry_fee.
        entry_cost = entry_trade.qty * entry_trade.price + entry_trade.fee
        exit_value = net_proceeds
        gross_pnl = (position.qty * fill_price) - (position.qty * position.entry_price)
        net_pnl = exit_value - entry_cost
        total_fees = entry_trade.fee + fee
        total_slippage = entry_trade.slippage + (position.qty * position.entry_price * self.slippage_rate)
        pnl_pct = net_pnl / entry_cost if entry_cost > 0 else 0.0

        # Trade на выход = продажа.
        exit_trade = Trade(
            position_id=position.id, side=TradeSide.sell,
            qty=position.qty, price=fill_price,
            fee=fee, slippage=total_slippage - entry_trade.slippage,
            pnl=net_pnl, reason=reason,
        )
        db.add(exit_trade)

        result = ClosedPositionResult(
            position=position, entry_trade=entry_trade, exit_trade=exit_trade,
            gross_pnl=gross_pnl, total_fees=total_fees,
            total_slippage=total_slippage, net_pnl=net_pnl,
            pnl_pct=pnl_pct, hold_bars=position.bars_held,
            exit_reason=reason.value,
        )
        logger.info(
            "ЗАКРЫТА позиция #%s %s reason=%s net_pnl=$%.4f (%.2f%%) hold=%d "
            "fee=$%.4f cash=$%.2f",
            position.id, coin.symbol, reason.value, net_pnl, pnl_pct * 100,
            position.bars_held, total_fees, session.cash,
        )
        return result

    async def _close_short(
        self,
        db: AsyncSession,
        session: Session,
        position: Position,
        coin: Coin,
        current_price: float,
        reason: TradeReason,
        entry_trade: Trade,
    ) -> ClosedPositionResult | None:
        """Закрыть SHORT: выкупаем по рынку (buy).

        PnL = (entry_price − fill_price) * qty. Возвращаем обеспечение ± PnL.
        """
        # Исполнение: выкупаем дороже из-за slippage.
        fill_price = current_price * (1.0 + self.slippage_rate)
        gross_value = position.qty * fill_price
        fee = gross_value * self.commission_rate

        # PnL short: прибыль при падении цены (entry > fill).
        gross_pnl = (position.entry_price - fill_price) * position.qty
        # Чистый PnL с учётом комиссий входа+выхода.
        net_pnl = gross_pnl - entry_trade.fee - fee
        # Возвращаем обеспечение + чистый PnL.
        # Обеспечение = entry_trade.qty * entry_trade.price (≈ position_value).
        collateral = entry_trade.qty * entry_trade.price
        session.cash += collateral + net_pnl

        # Закрываем позицию.
        position.status = PositionStatus.closed

        total_fees = entry_trade.fee + fee
        # Slippage short: на входе (продажа дешевле) + на выходе (выкуп дороже).
        exit_slippage = position.qty * current_price * self.slippage_rate
        total_slippage = entry_trade.slippage + exit_slippage
        pnl_pct = net_pnl / collateral if collateral > 0 else 0.0

        # Trade на выход short = покупка (выкуп).
        exit_trade = Trade(
            position_id=position.id, side=TradeSide.buy,
            qty=position.qty, price=fill_price,
            fee=fee, slippage=exit_slippage,
            pnl=net_pnl, reason=reason,
        )
        db.add(exit_trade)

        result = ClosedPositionResult(
            position=position, entry_trade=entry_trade, exit_trade=exit_trade,
            gross_pnl=gross_pnl, total_fees=total_fees,
            total_slippage=total_slippage, net_pnl=net_pnl,
            pnl_pct=pnl_pct, hold_bars=position.bars_held,
            exit_reason=reason.value,
        )
        logger.info(
            "ЗАКРЫТ SHORT #%s %s reason=%s net_pnl=$%.4f (%.2f%%) hold=%d "
            "fee=$%.4f cash=$%.2f",
            position.id, coin.symbol, reason.value, net_pnl, pnl_pct * 100,
            position.bars_held, total_fees, session.cash,
        )
        return result

    # ─── Live-ветка: реальные ордера через биржу ────────────

    async def _get_live_broker(self):
        """Создать LiveBroker из активного кошелька или None (нет кошелька)."""
        from core.wallet_service import get_default_wallet, get_decrypted_secret
        from core.market_service import LiveBroker
        from db.database import get_session

        async with get_session() as wdb:
            wallet = await get_default_wallet(wdb)
            if wallet is None:
                return None
            # Копируем нужные поля, пока сессия открыта.
            return LiveBroker(
                exchange_name=wallet.exchange,
                api_key=wallet.api_key,
                api_secret=get_decrypted_secret(wallet),
            )

    async def _open_position_live(
        self,
        db: AsyncSession,
        session: Session,
        coin: Coin,
        sizing: PositionSizing,
        timeframe: str,
        reason: TradeReason,
    ) -> tuple[Position, TradeExecution] | None:
        """Открыть LONG реальным рыночным ордером (live-режим).

        Ставит market-buy на position_value USDT через LiveBroker. Цена/qty/fee
        берутся из ответа биржи (реальное исполнение, не симуляция).
        Cash синхронизируется с реальным балансом биржи после ордера.
        """
        if sizing.qty <= 0 or sizing.position_value <= 0:
            logger.warning("live open_position: некорректный sizing, пропуск")
            return None

        broker = await self._get_live_broker()
        if broker is None:
            logger.warning("live open_position: нет кошелька, пропуск")
            return None

        try:
            # Реальный рыночный ордер на position_value USDT.
            order = await broker.create_market_buy(coin.symbol, sizing.position_value)
        except Exception as e:
            logger.error("live open_position %s: ордер не исполнен: %s", coin.symbol, e)
            return None
        finally:
            await broker.close()

        if order.qty <= 0 or order.avg_price <= 0:
            logger.warning(
                "live open_position %s: ордер не исполнен (qty=0)", coin.symbol,
            )
            return None

        # Создаём позицию с РЕАЛЬНЫМИ параметрами исполнения.
        position = Position(
            session_id=session.id, coin_id=coin.id, tf=timeframe,
            side=PositionSide.long,
            qty=order.qty, entry_price=order.avg_price,
            stop_price=sizing.stop_price,
            target_price=sizing.target_price,
            risk_amount=sizing.risk_amount,
            bars_held=0, status=PositionStatus.open,
        )
        db.add(position)
        await db.flush()

        entry_trade = Trade(
            position_id=position.id, side=TradeSide.buy,
            qty=order.qty, price=order.avg_price,
            fee=order.fee, slippage=0.0,  # реальный slippage уже в avg_price
            pnl=0.0, reason=reason,
        )
        db.add(entry_trade)
        await db.flush()

        # Синхронизируем cash с реальным балансом биржи.
        await self._sync_live_cash(session, broker=None)

        execution = TradeExecution(
            trade=entry_trade, fee=order.fee,
            slippage_cost=0.0, net_value=order.cost,
        )
        logger.info(
            "LIVE ОТКРЫТА позиция #%s %s qty=%.8f @ $%.4f fee=$%.6f order=%s",
            position.id, coin.symbol, order.qty, order.avg_price,
            order.fee, order.order_id,
        )
        return position, execution

    async def _close_position_live(
        self,
        db: AsyncSession,
        session: Session,
        position: Position,
        coin: Coin,
        reason: TradeReason,
    ) -> ClosedPositionResult | None:
        """Закрыть LONG реальным рыночным ордером sell (live-режим)."""
        entry_trade = await self._get_entry_trade(db, position)
        if entry_trade is None:
            return None

        broker = await self._get_live_broker()
        if broker is None:
            logger.warning("live close_position: нет кошелька, пропуск")
            return None

        try:
            order = await broker.create_market_sell(coin.symbol, position.qty)
        except Exception as e:
            logger.error("live close_position %s: ордер не исполнен: %s", coin.symbol, e)
            return None
        finally:
            await broker.close()

        if order.qty <= 0 or order.avg_price <= 0:
            logger.warning(
                "live close_position %s: ордер не исполнен (qty=0)", coin.symbol,
            )
            return None

        # Закрываем позицию.
        position.status = PositionStatus.closed

        # PnL по реальному исполнению.
        entry_cost = entry_trade.qty * entry_trade.price + entry_trade.fee
        exit_value = order.cost - order.fee
        gross_pnl = (order.avg_price - position.entry_price) * position.qty
        net_pnl = exit_value - entry_cost
        total_fees = entry_trade.fee + order.fee
        pnl_pct = net_pnl / entry_cost if entry_cost > 0 else 0.0

        exit_trade = Trade(
            position_id=position.id, side=TradeSide.sell,
            qty=order.qty, price=order.avg_price,
            fee=order.fee, slippage=0.0,
            pnl=net_pnl, reason=reason,
        )
        db.add(exit_trade)

        # Синхронизируем cash.
        await self._sync_live_cash(session, broker=None)

        result = ClosedPositionResult(
            position=position, entry_trade=entry_trade, exit_trade=exit_trade,
            gross_pnl=gross_pnl, total_fees=total_fees,
            total_slippage=0.0, net_pnl=net_pnl,
            pnl_pct=pnl_pct, hold_bars=position.bars_held,
            exit_reason=reason.value,
        )
        logger.info(
            "LIVE ЗАКРЫТА позиция #%s %s reason=%s net_pnl=$%.4f (%.2f%%) "
            "fee=$%.6f order=%s",
            position.id, coin.symbol, reason.value, net_pnl, pnl_pct * 100,
            total_fees, order.order_id,
        )
        return result

    async def _sync_live_cash(self, session: Session, broker=None) -> float:
        """Синхронизировать session.cash с реальным балансом USDT биржи.

        В live-режиме cash — это зеркало реального свободного USDT на бирже,
        а не локальный расчёт. Вызывается после каждой операции.
        """
        own_broker = broker is None
        if broker is None:
            broker = await self._get_live_broker()
        if broker is None:
            return session.cash
        try:
            usdt = await broker.fetch_usdt_balance()
            session.cash = usdt
            logger.info("LIVE cash синхронизирован с биржей: $%.2f", usdt)
            return usdt
        except Exception as e:
            logger.warning("LIVE sync cash не удался: %s", e)
            return session.cash
        finally:
            if own_broker:
                await broker.close()

    # ─── Equity ──────────────────────────────────────────────

    async def record_equity(
        self,
        db: AsyncSession,
        session: Session,
        prices: dict[int, float] | None = None,
    ) -> EquityPoint:
        """Записать точку equity-кривой.

        Args:
            prices: {coin_id: current_price} для расчёта нереализованного PnL.
                    Если None — только cash.

        Модель equity (paper trading):

            LONG  equity = cash + Σ(qty * px)               — актив растёт с ценой
            SHORT equity = cash + Σ(qty * entry + uPnL)     — обеспечение + нереализ.

        Где uPnL_short = qty * (entry − px) — прибыль шорта при падении цены.

        NB: для long стоимость позиции берётся по ТЕКУЩЕЙ рыночной цене px.
        При открытии сделки px ≈ entry_price, и equity сразу после входа ≈
        стартовый баланс − round-trip комиссия входа (fee входа уже списан
        из cash). Это правильное поведение: открытие позиции стоит комиссию.
        Асимметрии slippage нет, т.к. px = рыночная close (а не fill).
        """
        positions_value = 0.0
        if prices:
            # Суммируем рыночную стоимость открытых позиций.
            open_positions = await self.get_open_positions(db, session)
            for pos in open_positions:
                px = prices.get(pos.coin_id)
                # Если цены нет в словаре — используем entry_price как fallback
                # (лучше приблизительно, чем px=0, что удваивает short и
                # обнуляет long).
                if px is None:
                    px = pos.entry_price
                if pos.side == PositionSide.short:
                    # Обеспечение + нереализованный PnL (положителен при падении px).
                    positions_value += pos.qty * pos.entry_price + pos.qty * (pos.entry_price - px)
                else:
                    # LONG: оценка по текущей рыночной цене.
                    positions_value += pos.qty * px

        equity = session.cash + positions_value
        point = EquityPoint(
            session_id=session.id, equity=equity,
            cash=session.cash, positions_value=positions_value,
        )
        db.add(point)
        return point

    async def get_equity_curve(
        self, db: AsyncSession, session: Session, limit: int = 500,
    ) -> list[EquityPoint]:
        """Получить equity-кривую (хронологически)."""
        stmt = (
            select(EquityPoint)
            .where(EquityPoint.session_id == session.id)
            .order_by(EquityPoint.timestamp.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        return list(reversed(rows))

    # ─── Метрики аналитики ───────────────────────────────────

    async def compute_analytics(
        self, db: AsyncSession, session: Session,
    ) -> AnalyticsReport:
        """Посчитать сводные метрики (план 4.7).

        Учитывает и LONG, и SHORT позиции. Exit-trade определяется по стороне
        позиции (long → sell, short → buy), а не по глобальному фильтру side.
        """
        # Все сделки сессии, сгруппированные по позициям (entry + exit).
        stmt = (
            select(Trade)
            .join(Position, Trade.position_id == Position.id)
            .where(Position.session_id == session.id)
            .order_by(Position.id, Trade.executed_at)
        )
        result = await db.execute(stmt)
        all_trades = result.scalars().all()

        # Группируем по position_id: [entry_trade, exit_trade].
        from collections import defaultdict
        by_pos: dict[int, list[Trade]] = defaultdict(list)
        for t in all_trades:
            by_pos[t.position_id].append(t)

        exit_trades: list[Trade] = []
        entry_fees: list[float] = []
        for pos_id, trades in by_pos.items():
            if len(trades) < 2:
                # Позиция ещё не закрыта (только entry) — берём её комиссию входа.
                entry_fees.append(trades[0].fee)
                continue
            # entry = первая (хронологически), exit = вторая.
            entry_tr, exit_tr = trades[0], trades[1]
            exit_trades.append(exit_tr)
            entry_fees.append(entry_tr.fee)

        pnls = [t.pnl for t in exit_trades]
        fees = [t.fee for t in exit_trades]
        total_fees = sum(fees) + sum(entry_fees)

        total_trades = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        win_rate = len(wins) / total_trades if total_trades else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )
        total_pnl = sum(pnls)
        total_pnl_pct = total_pnl / session.initial_balance if session.initial_balance else 0.0
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0
        expectancy = (
            win_rate * avg_win - (1 - win_rate) * avg_loss
            if total_trades else 0.0
        )

        # Drawdown и Sharpe из equity-кривой.
        equity_curve = await self.get_equity_curve(db, session, limit=5000)
        max_dd = _max_drawdown([p.equity for p in equity_curve])
        rets = _equity_returns([p.equity for p in equity_curve])
        sharpe = _sharpe(rets)
        sortino = _sortino(rets)

        fee_drag = (total_fees / abs(total_pnl) * 100) if total_pnl != 0 else 0.0

        return AnalyticsReport(
            total_trades=total_trades, win_rate=win_rate,
            profit_factor=profit_factor, total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct, avg_win=avg_win, avg_loss=avg_loss,
            expectancy=expectancy, max_drawdown=max_dd,
            sharpe=sharpe, sortino=sortino, total_fees=total_fees,
            fee_drag_pct=fee_drag,
        )

    # ─── Чтение позиций ──────────────────────────────────────

    async def get_open_positions(
        self, db: AsyncSession, session: Session,
    ) -> list[Position]:
        stmt = (
            select(Position)
            .where(
                Position.session_id == session.id,
                Position.status == PositionStatus.open,
            )
            .order_by(Position.entry_at.desc())
        )
        return list((await db.execute(stmt)).scalars().all())

    async def get_open_position(
        self, db: AsyncSession, session: Session, coin: Coin,
    ) -> Position | None:
        stmt = (
            select(Position)
            .where(
                Position.session_id == session.id,
                Position.coin_id == coin.id,
                Position.status == PositionStatus.open,
            )
        )
        return (await db.execute(stmt)).scalars().first()

    async def get_open_positions_by_coin(
        self, db: AsyncSession, session: Session, coin: Coin,
    ) -> list[Position]:
        """Все открытые позиции по конкретной монете (для pyramiding-проверок)."""
        stmt = (
            select(Position)
            .where(
                Position.session_id == session.id,
                Position.coin_id == coin.id,
                Position.status == PositionStatus.open,
            )
            .order_by(Position.entry_at.desc())
        )
        return list((await db.execute(stmt)).scalars().all())

    async def get_last_closed_position_by_coin(
        self, db: AsyncSession, session: Session, coin: Coin,
    ) -> Position | None:
        """Самая свежая закрытая позиция по монете (для post-exit cooldown).

        Используется чтобы не открывать новую позицию сразу после закрытия
        (churn): закрыли по time_stop и тут же открыли новую по тому же
        сигналу — это сжигает cash на комиссиях без выгоды.
        """
        stmt = (
            select(Position)
            .where(
                Position.session_id == session.id,
                Position.coin_id == coin.id,
                Position.status == PositionStatus.closed,
            )
            .order_by(Position.id.desc())
            .limit(1)
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    async def _get_entry_trade(
        self, db: AsyncSession, position: Position,
    ) -> Trade | None:
        """Trade открытия позиции. Для long это buy, для short это sell.
        Берём самую раннюю сделку позиции (надёжнее фильтра по side).
        """
        stmt = (
            select(Trade)
            .where(Trade.position_id == position.id)
            .order_by(Trade.executed_at.asc())
            .limit(1)
        )
        return (await db.execute(stmt)).scalars().first()

    async def get_recent_trades(
        self, db: AsyncSession, session: Session, limit: int = 50,
    ) -> list[Trade]:
        """Последние сделки для UI-таблицы."""
        stmt = (
            select(Trade)
            .join(Position, Trade.position_id == Position.id)
            .where(Position.session_id == session.id)
            .order_by(Trade.executed_at.desc())
            .limit(limit)
        )
        return list((await db.execute(stmt)).scalars().all())


# ─── Вспомогательные функции ─────────────────────────────────


def sizing_qty_to_fill_price(sizing: PositionSizing, side: Literal["buy", "sell"]) -> float:
    """Цена исполнения с учётом slippage. Покупаем дороже, продаём дешевле."""
    from config import settings
    if side == "buy":
        return sizing.position_value / sizing.qty * (1.0 + settings.slippage_rate)
    raise NotImplementedError


def _max_drawdown(equity: list[float]) -> float:
    """Максимальная просадка equity-кривой (доля от пика)."""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _equity_returns(equity: list[float]) -> list[float]:
    """Доходности между точками equity."""
    rets = []
    for prev, curr in zip(equity, equity[1:]):
        if prev > 0:
            rets.append(curr / prev - 1.0)
    return rets


def _sharpe(returns: list[float], periods_per_year: int = 252, rf: float = 0.03) -> float:
    """Годовой Sharpe ratio. returns — дневные доходности."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    rf_per = rf / periods_per_year
    return (mean - rf_per) / std * math.sqrt(periods_per_year)


def _sortino(returns: list[float], periods_per_year: int = 252, rf: float = 0.03) -> float:
    """Годовой Sortino (учитывает только downside volatility)."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    downside = [min(0, r) for r in returns]
    var = sum(d ** 2 for d in downside) / len(downside)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    rf_per = rf / periods_per_year
    return (mean - rf_per) / std * math.sqrt(periods_per_year)


# ─── Синглтон ────────────────────────────────────────────────

_portfolio: PortfolioEngine | None = None


def get_portfolio() -> PortfolioEngine:
    global _portfolio
    if _portfolio is None:
        _portfolio = PortfolioEngine()
    return _portfolio
