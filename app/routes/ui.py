"""Роуты UI: дашборд, выбор режима, экран монеты, аналитика, настройки.

Все шаблоны — Jinja2 + HTMX (server-rendered, как у Kronos).
Plotly графики генерируются сервером и рендерятся как JSON в <div id="chart">.

Символы пар (BTC/USDT) в URL кодируются как BTC-USDT (заменяем / на -),
чтобы FastAPI не воспринимал слэш как разделитель пути.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select

from config import settings
from core.data_lifecycle import get_dlm
from core.indicators import compute_all
from core.portfolio_engine import get_portfolio
from core.prediction_service import get_prediction_service
from core.trading_loop import get_loop
from core.mode_manager import get_active_mode
from db.database import get_session
from db.models import Coin, EquityPoint, SessionMode, Position, Trade

logger = logging.getLogger("trading.ui")

router = APIRouter()


# ─── Кодирование символов для URL ──────────────────────────────

def _encode_symbol(symbol: str) -> str:
    """BTC/USDT → BTC-USDT для использования в URL-пути."""
    return symbol.replace("/", "-")


def _decode_symbol(encoded: str) -> str:
    """BTC-USDT → BTC/USDT (обратное преобразование)."""
    return encoded.replace("-", "/")


# ─── Экран выбора режима ─────────────────────────────────────

@router.get("/")
async def index(request: Request) -> HTMLResponse:
    """Главная: редирект на дашборд."""
    return RedirectResponse(url="/dashboard", status_code=303)


@router.get("/dashboard")
async def dashboard(request: Request) -> HTMLResponse:
    """Дашборд: equity-кривая, баланс, список монет."""
    from app.main import TEMPLATES

    mode = await get_active_mode()
    async with get_session() as db:
        portfolio = get_portfolio()
        session = await portfolio.get_or_create_session(db, mode)
        report = await portfolio.compute_analytics(db, session)
        equity_curve = await portfolio.get_equity_curve(db, session, limit=200)
        positions_raw = await portfolio.get_open_positions(db, session)
        coins = (await db.execute(select(Coin).where(Coin.enabled == True))).scalars().all()
        # Явно подгружаем coin для каждой позиции (иначе detached → '?' в шаблоне).
        # Собираем {coin_id: symbol} одним запросом.
        coin_ids = {p.coin_id for p in positions_raw}
        coin_map: dict[int, str] = {}
        if coin_ids:
            coin_rows = (await db.execute(
                select(Coin.id, Coin.symbol).where(Coin.id.in_(coin_ids))
            )).all()
            coin_map = {row[0]: row[1] for row in coin_rows}
        positions = [
            {
                "id": p.id,
                "coin_symbol": coin_map.get(p.coin_id, "?"),
                "side": p.side,
                "tf": p.tf,
                "qty": p.qty,
                "entry_price": p.entry_price,
                "stop_price": p.stop_price,
                "target_price": p.target_price,
                "bars_held": p.bars_held,
            }
            for p in positions_raw
        ]

    # Данные для Plotly: equity curve.
    eq_x = [p.timestamp.isoformat() for p in equity_curve] if equity_curve else []
    eq_y = [p.equity for p in equity_curve] if equity_curve else [100.0]
    # Данные для Plotly: drawdown.
    dd_y = _drawdown_series(eq_y) if len(eq_y) > 1 else []

    # Монеты с текущими ценами и статусами auto.
    coin_data = []
    for coin in coins:
        coin_data.append({
            "symbol": coin.symbol,
            "encoded": _encode_symbol(coin.symbol),
            "default_tf": coin.default_tf,
            "auto_4h": get_loop().is_auto(coin.symbol, "4h"),
            "auto_1h": get_loop().is_auto(coin.symbol, "1h"),
        })

    return TEMPLATES.TemplateResponse("dashboard.html", {
        "request": request,
        "session": session,
        "report": report,
        "coins": coin_data,
        "positions": positions,
        "eq_x": eq_x,
        "eq_y": eq_y,
        "dd_y": dd_y,
        "last_equity": eq_y[-1] if eq_y else 100.0,
        "initial_equity": session.initial_balance,
    })


@router.get("/coin/{encoded_symbol}")
async def coin_page(request: Request, encoded_symbol: str, tf: str = "4h") -> HTMLResponse:
    """Экран монеты: график свечей + прогноз Kronos, индикаторы, сигналы, кнопки."""
    from app.main import TEMPLATES

    symbol = _decode_symbol(encoded_symbol)
    loop = get_loop()
    async with get_session() as db:
        portfolio = get_portfolio()
        session = await portfolio.get_or_create_session(db, await get_active_mode())
        coin = (await db.execute(
            select(Coin).where(Coin.symbol == symbol)
        )).scalar_one_or_none()
        if coin is None:
            return HTMLResponse("Монета не найдена", status_code=404)

        # Открытые сделки по этой монете (для блока «Открытые сделки» в UI).
        open_positions_raw = await portfolio.get_open_positions_by_coin(db, session, coin)
        open_positions = [
            {
                "id": p.id,
                "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                "tf": p.tf,
                "qty": p.qty,
                "entry_price": p.entry_price,
                "stop_price": p.stop_price,
                "target_price": p.target_price,
                "bars_held": p.bars_held,
                "entry_at": p.entry_at,
            }
            for p in open_positions_raw
        ]

        # Запускаем тик: обновляем данные, считаем всё.
        tick = await loop.run_once(symbol, tf)

        # Получаем свечи для графика + мини-бэктеста (512 контекст + 16 шагов).
        dlm = get_dlm()
        candles = await dlm.get_candles(db, coin.id, tf, limit=settings.data_kronos_context + 30)

        # Графику показываем последние 200 свечей (данных для модели не трогаем — они отдельно).
        # Отбрасываем последнюю (формирующуюся) свечу: она ещё не закрыта, и прогноз
        # Kronos начинается со следующей за ней. Иначе мост прогноза визуально
        # «накладывается» на формирующуюся свечу — выглядит как смещение.
        chart_candles_all = candles[-201:] if len(candles) > 201 else candles
        chart_candles = chart_candles_all[:-1] if len(chart_candles_all) > 1 else chart_candles_all

        # Индикаторы на контексте.
        import pandas as pd
        if len(chart_candles) >= 20:
            df = pd.DataFrame({
                "timestamp": [c.timestamp for c in chart_candles],
                "open": [c.open for c in chart_candles],
                "high": [c.high for c in chart_candles],
                "low": [c.low for c in chart_candles],
                "close": [c.close for c in chart_candles],
                "volume": [c.volume for c in chart_candles],
            })
            snap = compute_all(
                df["timestamp"], df["close"], df["high"], df["low"], df["volume"]
            )
        else:
            snap = None

        # Kronos прогноз (если есть).
        pred_svc = get_prediction_service()
        pred = None
        if pred_svc.is_loaded and len(candles) >= settings.data_kronos_context:
            pred = await asyncio.to_thread(
                lambda: pred_svc.predict(candles, symbol, tf)
            )

        # Мини-бэктест точности (если Kronos загружен и хватает данных).
        accuracy = []
        if pred_svc.is_loaded and len(candles) >= settings.data_kronos_context + 16:
            accuracy = await asyncio.to_thread(
                lambda: pred_svc.mini_backtest(candles, symbol, tf, steps=16)
            )

    # Данные для Plotly: свечи (последние 200) + прогноз + точность.
    chart_ohlc = _ohlc_for_plotly(chart_candles) if chart_candles else []
    chart_pred_close = pred.pred_close if pred else []
    chart_pred_high = pred.pred_high if pred else []
    chart_pred_low = pred.pred_low if pred else []
    chart_timestamps = _timestamps_for_plotly(chart_candles) if chart_candles else []
    # Точность — только для последних 16 свечей (привязка к timestamps графика).
    chart_accuracy = _accuracy_for_plotly(chart_candles, accuracy)

    return TEMPLATES.TemplateResponse("coin.html", {
        "request": request,
        "session": session,
        "symbol": symbol,
        "encoded": encoded_symbol,
        "tf": tf,
        "tick": tick,
        "tick_s": loop.get_last_signal(symbol, tf) or {},
        "snap": snap,
        "pred": pred,
        "auto_enabled": loop.is_auto(symbol, tf),
        "entry_threshold": settings.entry_threshold,
        "strategy_min_adx": settings.strategy_min_adx,
        "strategy_min_rr": settings.strategy_min_rr,
        "accuracy_summary": _accuracy_summary(accuracy),
        "open_positions": open_positions,
        "current_price": (tick.close if tick and tick.close else 0.0),
        "atr": (snap.atr if snap and snap.atr else 0.0),
        "atr_k_stop": settings.atr_k_stop,
        "atr_k_take": settings.atr_k_take,
        # Данные для Plotly (передаём как JSON в шаблон).
        "chart_ohlc": chart_ohlc,
        "chart_pred_close": chart_pred_close,
        "chart_pred_high": chart_pred_high,
        "chart_pred_low": chart_pred_low,
        "chart_timestamps": chart_timestamps,
        "chart_accuracy": chart_accuracy,
        # Новостной индикатор (LLM + Search).
        "news_enabled": settings.news_enabled,
        "news_has_key": (
            (settings.news_provider == "groq" and bool(settings.groq_api_key))
            or (settings.news_provider == "openrouter" and bool(settings.openrouter_api_key))
            or (settings.news_provider == "gemini" and bool(settings.gemini_api_key))
        ),
        "news_provider": settings.news_provider,
        "news_lookback_days": settings.news_lookback_days,
    })


@router.post("/coin/{encoded_symbol}/tick")
async def manual_tick(request: Request, encoded_symbol: str, tf: str = "4h"):
    """HTMX: ручной тик по монете (кнопка «Обновить»)."""
    symbol = _decode_symbol(encoded_symbol)
    loop = get_loop()
    result = await loop.run_once(symbol, tf)
    import json as j
    return HTMLResponse(
        f'<div class="card"><small>ТИК: S_entry={result.s_entry:.3f} '
        f'action={result.action} {result.detail}</small></div>'
    )


@router.post("/coin/{encoded_symbol}/auto")
async def toggle_auto(request: Request, encoded_symbol: str, tf: str = "4h"):
    """HTMX: переключить авто-режим.

    Возвращает элемент с hx-атрибутами для дальнейших переключений.
    Кнопка содержит data-sym/data-tf/auto-btn — чтобы bulk-toggle JS
    мог найти её после HTMX-swap.
    """
    symbol = _decode_symbol(encoded_symbol)
    loop = get_loop()
    enabled = not loop.is_auto(symbol, tf)
    loop.set_auto(symbol, tf, enabled)
    cls = "btn btn-sm auto-btn" + (" btn-success" if enabled else "")
    state = "ON" if enabled else "OFF"
    return HTMLResponse(
        f'<button type="button" class="{cls}" '
        f'data-sym="{symbol}" data-tf="{tf}" '
        f'hx-post="/coin/{encoded_symbol}/auto?tf={tf}" '
        f'hx-target="this" hx-swap="outerHTML" '
        f'title="Авто-торговля {tf}">'
        f'{state}'
        f'</button>'
    )


@router.post("/coin/{encoded_symbol}/buy")
async def manual_buy(request: Request, encoded_symbol: str):
    """HTMX: ручная покупка (кнопка «Купить»)."""
    symbol = _decode_symbol(encoded_symbol)
    loop = get_loop()
    loop.set_auto(symbol, "4h", True)
    tick = await loop.run_once(symbol, "4h")
    loop.set_auto(symbol, "4h", False)
    return HTMLResponse(f'<small>ТИК: {tick.action} {tick.detail}</small>')


@router.post("/coin/{encoded_symbol}/sell")
async def manual_sell(request: Request, encoded_symbol: str):
    """HTMX: ручная продажа."""
    symbol = _decode_symbol(encoded_symbol)
    from db.database import get_session
    from core.portfolio_engine import get_portfolio
    from db.models import Coin, PositionStatus, TradeReason
    from core.market_service import get_market
    from core.data_lifecycle import get_dlm

    async with get_session() as db:
        portfolio = get_portfolio()
        session = await portfolio.get_or_create_session(db, await get_active_mode())
        coin = (await db.execute(select(Coin).where(Coin.symbol == symbol))).scalar_one_or_none()
        if coin is None:
            return HTMLResponse("Монета не найдена", status_code=404)

        pos = await portfolio.get_open_position(db, session, coin)
        if pos is None:
            return HTMLResponse("<small>Нет открытой позиции</small>")

        dlm = get_dlm()
        candles = await dlm.get_candles(db, coin.id, "4h", limit=5)
        price = candles[-1].close if candles else pos.entry_price
        closed = await portfolio.close_position(
            db, session, pos, coin, price, TradeReason.manual
        )
        # Записываем equity сразу после закрытия — иначе дашборд покажет
        # устаревший баланс (следующий equity_point только через 15 мин).
        # flush ОБЯЗАТЕЛЕН: иначе при autoflush=False get_open_positions в
        # record_equity всё ещё видит закрытую позицию как открытую.
        if closed:
            await db.flush()
            await portfolio.record_equity(db, session, prices={coin.id: price})
        await db.commit()
        if closed:
            return HTMLResponse(
                f'<small>ЗАКРЫТА: {closed.exit_reason} PnL=${closed.net_pnl:.4f}</small>'
            )
    return HTMLResponse("<small>Ошибка</small>")


# ─── Ручная торговая панель (модалка Long/Short + qty + price) ──


@router.post("/coin/{encoded_symbol}/manual-open")
async def manual_open(request: Request, encoded_symbol: str, tf: str = "4h"):
    """HTMX: открыть сделку вручную из модалки (сторона + qty + цена).

    Форма отправляет: side (long|short), qty, price. Стопы считаются из ATR.
    Возвращает HTML-сниппет с результатом для #manual-result.
    """
    symbol = _decode_symbol(encoded_symbol)
    from db.database import get_session
    from core.portfolio_engine import get_portfolio
    from core.data_lifecycle import get_dlm
    from core.indicators import compute_all
    from db.models import Coin, TradeReason
    import pandas as pd

    form = await request.form()
    side = str(form.get("side", "long")).lower()
    try:
        qty = float(form.get("qty", 0))
        price = float(form.get("price", 0))
        stop_price = float(form.get("stop_price", 0))
        target_price = float(form.get("target_price", 0))
    except (ValueError, TypeError):
        return HTMLResponse('<small style="color: var(--red);">Некорректные значения</small>')

    if side not in ("long", "short"):
        return HTMLResponse('<small style="color: var(--red);">Сторона должна быть long/short</small>')
    if qty <= 0 or price <= 0:
        return HTMLResponse('<small style="color: var(--red);">qty и price должны быть &gt; 0</small>')

    async with get_session() as db:
        portfolio = get_portfolio()
        mode = await get_active_mode()
        session = await portfolio.get_or_create_session(db, mode)
        coin = (await db.execute(select(Coin).where(Coin.symbol == symbol))).scalar_one_or_none()
        if coin is None:
            return HTMLResponse("Монета не найдена", status_code=404)

        # ATR для расчёта стопов (если данных достаточно).
        dlm = get_dlm()
        atr = None
        try:
            candles = await dlm.get_candles(db, coin.id, tf, limit=250)
            if len(candles) >= 20:
                df = pd.DataFrame({
                    "timestamp": [c.timestamp for c in candles],
                    "close": [c.close for c in candles],
                    "high": [c.high for c in candles],
                    "low": [c.low for c in candles],
                    "volume": [c.volume for c in candles],
                })
                snap = compute_all(
                    df["timestamp"], df["close"], df["high"], df["low"], df["volume"]
                )
                atr = snap.atr
        except Exception:
            atr = None

        result = await portfolio.open_manual_position(
            db, session, coin, side, qty, price, tf, atr=atr,
            stop_price=stop_price if stop_price > 0 else None,
            target_price=target_price if target_price > 0 else None,
            reason=TradeReason.manual,
        )
        # Записываем equity сразу после открытия.
        if result:
            await db.flush()
            await portfolio.record_equity(db, session, prices={coin.id: price})
        await db.commit()

    if result is None:
        return HTMLResponse(
            '<small style="color: var(--red);">Не удалось открыть сделку '
            '(недостаточно средств или шорт недоступен в live)</small>'
        )
    position = result[0]
    side_ru = "LONG" if side == "long" else "SHORT"
    color = "var(--green)" if side == "long" else "var(--red)"
    return HTMLResponse(
        f'<small style="color: {color};">✅ {side_ru} #{position.id} открыт: '
        f'qty={qty:.6f} @ ${price:.2f}</small>'
    )


@router.post("/positions/{position_id}/close")
async def close_position_by_id(request: Request, position_id: int):
    """HTMX: закрыть конкретную открытую сделку из списка по её id.

    Возвращает HTML-сниппет с результатом.
    """
    from db.database import get_session
    from core.portfolio_engine import get_portfolio
    from core.data_lifecycle import get_dlm
    from db.models import Coin, Position, PositionStatus, TradeReason

    async with get_session() as db:
        portfolio = get_portfolio()
        mode = await get_active_mode()
        session = await portfolio.get_or_create_session(db, mode)
        pos = (await db.execute(
            select(Position).where(
                Position.id == position_id,
                Position.session_id == session.id,
                Position.status == PositionStatus.open,
            )
        )).scalar_one_or_none()
        if pos is None:
            return HTMLResponse('<small style="color: var(--red);">Сделка не найдена</small>')

        coin = (await db.execute(select(Coin).where(Coin.id == pos.coin_id))).scalar_one_or_none()
        if coin is None:
            return HTMLResponse('<small style="color: var(--red);">Монета не найдена</small>')

        # Текущая цена для закрытия.
        dlm = get_dlm()
        candles = await dlm.get_candles(db, coin.id, pos.tf or "4h", limit=5)
        price = candles[-1].close if candles else pos.entry_price

        closed = await portfolio.close_position(
            db, session, pos, coin, price, TradeReason.manual
        )
        # Записываем equity сразу после закрытия — иначе дашборд покажет
        # устаревший баланс (следующий equity_point только через 15 мин).
        # flush ОБЯЗАТЕЛЕН: иначе при autoflush=False get_open_positions в
        # record_equity всё ещё видит закрытую позицию как открытую (status
        # изменился в памяти, но не в БД) → positions_value завышается.
        if closed is not None:
            await db.flush()
            await portfolio.record_equity(db, session, prices={coin.id: price})
        await db.commit()

    if closed is None:
        return HTMLResponse('<small style="color: var(--red);">Не удалось закрыть</small>')
    pnl_color = "var(--green)" if closed.net_pnl >= 0 else "var(--red)"
    return HTMLResponse(
        f'<small style="color: {pnl_color};">✅ Закрыта #{position_id} '
        f'{closed.exit_reason} PnL=${closed.net_pnl:.4f}</small>'
    )


@router.get("/api/position-pnl")
async def position_pnl_api(request: Request, id: int):
    """Расчёт PnL открытой позиции по текущей цене (для модалки подтверждения).

    Возвращает: { pnl, price, side, qty, entry_price }
    """
    from db.database import get_session
    from core.portfolio_engine import get_portfolio
    from core.data_lifecycle import get_dlm
    from db.models import Position, PositionStatus, PositionSide

    async with get_session() as db:
        portfolio = get_portfolio()
        mode = await get_active_mode()
        session = await portfolio.get_or_create_session(db, mode)
        pos = (await db.execute(
            select(Position).where(
                Position.id == id,
                Position.session_id == session.id,
                Position.status == PositionStatus.open,
            )
        )).scalar_one_or_none()
        if pos is None:
            return {"pnl": 0, "price": 0, "side": "long", "qty": 0, "entry_price": 0}

        dlm = get_dlm()
        candles = await dlm.get_candles(db, pos.coin_id, pos.tf or "4h", limit=5)
        price = candles[-1].close if candles else pos.entry_price

        if pos.side == PositionSide.short:
            pnl = (pos.entry_price - price) * pos.qty
        else:
            pnl = (price - pos.entry_price) * pos.qty

    return {
        "pnl": round(pnl, 4),
        # price без round() — JS форматирует через pricefmtJS (важно для дешёвых монет).
        "price": price,
        "side": pos.side.value if hasattr(pos.side, "value") else str(pos.side),
        "qty": pos.qty,
        "entry_price": pos.entry_price,
    }


@router.post("/positions/{position_id}/edit")
async def edit_position(request: Request, position_id: int):
    """HTMX: изменить стоп-лосс и тейк-профит открытой позиции.

    Принимает stop_price, target_price из формы. Валидирует по стороне сделки:
      long  → stop < entry < target
      short → target < entry < stop
    Возвращает HTMX-сниппет с результатом.
    """
    from db.database import get_session
    from core.portfolio_engine import get_portfolio
    from db.models import Position, PositionStatus, PositionSide

    form = await request.form()

    def _parse(name: str) -> float | None:
        raw = form.get(name)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    new_stop = _parse("stop_price")
    new_target = _parse("target_price")

    async with get_session() as db:
        portfolio = get_portfolio()
        mode = await get_active_mode()
        session = await portfolio.get_or_create_session(db, mode)
        pos = (await db.execute(
            select(Position).where(
                Position.id == position_id,
                Position.session_id == session.id,
                Position.status == PositionStatus.open,
            )
        )).scalar_one_or_none()

        if pos is None:
            return HTMLResponse(
                '<small style="color: var(--red);">Сделка не найдена или уже закрыта</small>'
            )

        entry = pos.entry_price
        side = pos.side.value if hasattr(pos.side, "value") else str(pos.side)
        errors: list[str] = []

        # Применяем только переданные значения; пустое поле = не меняем.
        if new_stop is not None:
            if new_stop <= 0:
                errors.append("Стоп должен быть положительным")
            elif side == "short":
                if new_stop <= entry:
                    errors.append(f"Для SHORT стоп должен быть ВЫШЕ входа ({entry})")
            else:
                if new_stop >= entry:
                    errors.append(f"Для LONG стоп должен быть НИЖЕ входа ({entry})")

        if new_target is not None:
            if new_target <= 0:
                errors.append("Тейк должен быть положительным")
            elif side == "short":
                if new_target >= entry:
                    errors.append(f"Для SHORT тейк должен быть НИЖЕ входа ({entry})")
            else:
                if new_target <= entry:
                    errors.append(f"Для LONG тейк должен быть ВЫШЕ входа ({entry})")

        # Согласованность stop ↔ target, если заданы оба.
        if new_stop is not None and new_target is not None:
            eff_stop = new_stop if new_stop is not None else pos.stop_price
            eff_target = new_target if new_target is not None else pos.target_price
            if side == "short" and eff_target >= eff_stop:
                errors.append("Для SHORT тейк должен быть ниже стопа")
            elif side != "short" and eff_stop >= eff_target:
                errors.append("Для LONG стоп должен быть ниже тейка")

        if errors:
            msg = "⚠️ " + "; ".join(errors)
            return HTMLResponse(
                f'<small style="color: var(--red);">{msg}</small>',
                status_code=400,
            )

        # Применяем изменения.
        if new_stop is not None:
            pos.stop_price = new_stop
        if new_target is not None:
            pos.target_price = new_target
        await db.commit()

        stop_str = f"{pos.stop_price:.6g}" if pos.stop_price else "—"
        target_str = f"{pos.target_price:.6g}" if pos.target_price else "—"
        logger.info(
            "Позиция #%s (%s) обновлена: stop=%s target=%s",
            position_id, side.upper(), stop_str, target_str,
        )

    return HTMLResponse(
        f'<small style="color: var(--green);">✓ Обновлено: стоп {stop_str}, '
        f'тейк {target_str}</small>'
    )


@router.post("/coin/{encoded_symbol}/short")
async def manual_short(request: Request, encoded_symbol: str):
    """HTMX: ручное открытие шорта. Принудительно пробует short (сниженные пороги)."""
    symbol = _decode_symbol(encoded_symbol)
    from core.portfolio_engine import get_portfolio
    from core.prediction_service import get_prediction_service
    from core.data_lifecycle import get_dlm
    from core.indicators import compute_all
    from db.models import Coin, PositionStatus, TradeReason, PositionSide
    import pandas as pd
    import asyncio

    loop = get_loop()
    portfolio = get_portfolio()
    pred_svc = get_prediction_service()
    dlm = get_dlm()

    async with get_session() as db:
        session = await portfolio.get_or_create_session(db, await get_active_mode())
        coin = (await db.execute(select(Coin).where(Coin.symbol == symbol))).scalar_one_or_none()
        if coin is None:
            return HTMLResponse("Монета не найдена", status_code=404)

        # Проверяем нет ли уже открытой позиции.
        existing = await portfolio.get_open_position(db, session, coin)
        if existing is not None:
            return HTMLResponse("<small>Уже есть открытая позиция</small>")

        # Получаем данные для сигнала.
        await dlm.ensure_history(db, coin, "4h")
        candles = await dlm.get_candles(db, coin.id, "4h",
                                          limit=max(settings.data_kronos_context, settings.data_indicator_warmup) + 5)
        if len(candles) < settings.data_indicator_warmup:
            return HTMLResponse("<small>Мало данных для шорта</small>")

        df = pd.DataFrame({
            "timestamp": [c.timestamp for c in candles],
            "close": [c.close for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "volume": [c.volume for c in candles],
        })
        snap = compute_all(df["timestamp"], df["close"], df["high"], df["low"], df["volume"])

        pred = None
        if pred_svc.is_loaded:
            pred = await asyncio.to_thread(
                lambda: pred_svc.predict(candles, symbol, "4h")
            )

        # Считаем short-сигнал.
        signal = loop.strategy.compute_signal(
            snap,
            kronos_pred_return=pred.pred_return if pred else None,
            kronos_pred_slope=pred.pred_path_slope if pred else None,
            kronos_pred_max=pred.pred_max_high if pred else None,
            kronos_pred_min=pred.pred_min_low if pred else None,
        )

        # Ручной шорт — пробуем принудительно с пониженным порогом.
        should_short, sizing, _ = loop.strategy.evaluate_short_entry(
            snap, signal.s_entry_short, signal.sub_signals_short,
            equity=session.cash, cash=session.cash,
            kronos_pred_return=pred.pred_return if pred else None,
            kronos_pred_min=pred.pred_min_low if pred else None,
        )

        if not should_short or sizing is None:
            # Если фильтры не пропустили — открываем всё равно с sizing по умолчанию.
            risk_mgr = loop.strategy.risk_mgr
            try:
                sizing = risk_mgr.calculate_short(
                    entry_price=snap.close, atr=snap.atr,
                    equity=session.cash, cash=session.cash,
                    target_from_prediction=pred.pred_min_low if pred else None,
                )
            except (ValueError, ZeroDivisionError):
                return HTMLResponse("<small>Не удалось рассчитать sizing для шорта</small>")

        result = await portfolio.open_short_position(
            db, session, coin, sizing, "4h", TradeReason.manual,
            force=True,
        )
        await db.commit()

        if result:
            pos, execution = result
            return HTMLResponse(
                f'<small>SHORT ОТКРЫТ: {coin.symbol} qty={sizing.qty:.6f} '
                f'@ ${sizing.stop_price:.2f} stop / ${sizing.target_price:.2f} target</small>'
            )
    return HTMLResponse("<small>Ошибка открытия шорта</small>")


# ─── Live-данные (polling) ──────────────────────────────────────

# Кеш актуальной цены: {symbol: {"price": float, "fetched_at": float}}
# Обновляется не чаще раза в 15 сек, чтобы не упереться в rate-limit Binance.
_price_cache: dict[str, dict] = {}
_PRICE_TTL = 15.0  # секунд


@router.get("/api/live/{encoded_symbol}")
async def live_data(encoded_symbol: str, tf: str = "4h") -> dict:
    """Live-данные для polling: цена, вероятность, суб-сигналы.

    Опрашивается браузером каждые 5 сек. Чтение из кеша TradingLoop — мгновенно,
    цена обновляется с Binance не чаще раза в 15 сек (rate-limit friendly).
    """
    import time
    symbol = _decode_symbol(encoded_symbol)
    loop = get_loop()

    # Сигнал из кеша СТРОГО для выбранного ТФ.
    # Раньше был fallback на другой ТФ — это подменяло direction/win_prob и
    # вызывало баг «между 1h и 4h»: подпись бралась с чужого ТФ. Теперь сигнал
    # берём только своего ТФ; если его нет — отдаём null и фронт не трогает UI.
    sig = loop.get_last_signal(symbol, tf) or {}
    # Цена: если своего ТФ нет — берём с любого доступного (цена едина для монеты).
    if not sig:
        any_sig = loop.get_last_signal(symbol, "4h") or loop.get_last_signal(symbol, "1h") or {}
    else:
        any_sig = sig

    # Актуальная цена: из кеша или fetch_ticker (не чаще 15 сек).
    now = time.time()
    cached = _price_cache.get(symbol)
    live_price = any_sig.get("close")  # fallback на close последней свечи
    if cached and (now - cached["fetched_at"]) < _PRICE_TTL:
        live_price = cached["price"]
    else:
        try:
            market = get_market()
            ticker = await market.fetch_ticker(symbol)
            live_price = ticker["last"] or live_price
            _price_cache[symbol] = {"price": live_price, "fetched_at": now}
        except Exception:
            # Binance недоступен — отдаём последнюю известную.
            if cached:
                live_price = cached["price"]

    return {
        "symbol": symbol,
        "price": live_price,
        # s_entry/direction/win_prob — ТОЛЬКО из своего ТФ (None если нет данных).
        "s_entry": sig.get("s_entry"),
        "s_entry_short": sig.get("s_entry_short"),
        "win_prob": sig.get("win_prob"),
        "direction": sig.get("direction"),
        "sub_signals": sig.get("sub_signals", {}),
        "sub_signals_short": sig.get("sub_signals_short", {}),
        "action": sig.get("action", "—"),
        "pred_return": sig.get("pred_return"),
        "updated_at": sig.get("updated_at"),
        # Новости (Gemini + Google Search).
        "news_sentiment": sig.get("news_sentiment"),
        "news_summary": sig.get("news_summary"),
        "news_confidence": sig.get("news_confidence"),
        "news_sources": sig.get("news_sources", []),
        "news_key_events": sig.get("news_key_events", []),
        "news_fetched_at": sig.get("news_fetched_at"),
    }


# ─── Новости: принудительное обновление ──────────────────────────


@router.post("/api/news/{encoded_symbol}/refresh")
async def refresh_news(encoded_symbol: str) -> dict:
    """Принудительно обновить новости по символу (сброс кеша + свежий запрос).

    Провайдер определяется настройкой news_provider:
      - "openrouter": OpenRouter LLM + Tavily Search (дефолт, бесплатно).
      - "gemini": Gemini + Google Search (legacy).

    Вызывается кнопкой «↻ Обновить» на карточке новостей.
    Возвращает тот же формат, что /api/live по части news_*.
    """
    symbol = _decode_symbol(encoded_symbol)
    if not settings.news_enabled:
        return {"ok": False, "error": "Новости отключены (NEWS_ENABLED=false)"}

    # Проверка ключей в зависимости от провайдера.
    provider = settings.news_provider
    if provider == "groq" and not settings.groq_api_key:
        return {"ok": False, "error": "Нет GROQ_API_KEY. Получи ключ: console.groq.com/keys"}
    if provider == "openrouter" and not settings.openrouter_api_key:
        return {"ok": False, "error": "Нет OPENROUTER_API_KEY. Получи ключ: openrouter.ai/keys"}
    if provider == "gemini" and not settings.gemini_api_key:
        return {"ok": False, "error": "Нет GEMINI_API_KEY. Получи ключ: aistudio.google.com/apikey"}

    from core.news_service import get_news_service
    news_svc = get_news_service()
    news_svc.clear_cache(symbol)
    try:
        result = await news_svc.get_sentiment(symbol, force=True)
    except Exception as exc:
        return {"ok": False, "error": f"Ошибка: {exc}"}

    if result is None:
        return {"ok": False, "error": "Не удалось получить новости"}

    return {
        "ok": True,
        "sentiment": result.sentiment,
        "summary": result.summary,
        "confidence": result.confidence,
        "sources": result.sources,
        "key_events": result.key_events,
        "fetched_at": result.fetched_at.isoformat(),
        "via_search": result.via_search,
    }


# ─── Дашборд: вероятности роста и фоновый пересчёт ────────────────


@router.get("/api/dashboard-signals")
async def dashboard_signals() -> dict:
    """Сводка вероятностей роста по всем монетам × ТФ (1h, 4h) из кеша.

    Чтение из _last_signals TradingLoop — мгновенно, без GPU/сети.
    Опрашивается дашбордом (polling каждые 20 сек) для отрисовки баров
    вероятности роста. Возвращает по каждой монете win_prob и направление
    (pred_return > 0 = рост) для обоих таймфреймов.
    """
    loop = get_loop()
    async with get_session() as db:
        coins = (await db.execute(select(Coin).where(Coin.enabled == True))).scalars().all()

    items: list[dict] = []
    for coin in coins:
        s4 = loop.get_last_signal(coin.symbol, "4h") or {}
        s1 = loop.get_last_signal(coin.symbol, "1h") or {}
        items.append({
            "symbol": coin.symbol,
            "encoded": _encode_symbol(coin.symbol),
            "default_tf": coin.default_tf,
            "win_prob_4h": s4.get("win_prob"),
            "pred_return_4h": s4.get("pred_return"),
            "direction_4h": s4.get("direction", "up"),
            "updated_4h": s4.get("updated_at"),
            "win_prob_1h": s1.get("win_prob"),
            "pred_return_1h": s1.get("pred_return"),
            "direction_1h": s1.get("direction", "up"),
            "updated_1h": s1.get("updated_at"),
        })
    return {
        "coins": items,
        "refreshing": loop.is_refreshing,
        "entry_threshold": settings.entry_threshold,
    }


@router.post("/api/recalc")
async def recalc_predictions():
    """HTMX: запустить фоновый пересчёт прогнозов по всем монетам × ТФ.

    Немедленно возвращает сниппет, а пересчёт идёт в фоне (asyncio.create_task).
    Бары вероятностей подтянутся на дашборде по polling по мере готовности.
    """
    loop = get_loop()
    if loop.is_refreshing:
        return HTMLResponse("<small>Пересчёт уже идёт…</small>")
    asyncio.create_task(loop.refresh_all_predictions())
    return HTMLResponse("<small>Пересчёт запущен…</small>")


@router.post("/api/auto-all")
async def auto_all(tf: str = "4h", enable: bool = True):
    """HTMX/JS: массовое включение/выключение авто-торговли для всех монет по ТФ.

    Возвращает JSON со списком затронутых символов — фронтенд обновляет кнопки
    в каждой строке без перезагрузки страницы.
    """
    loop = get_loop()
    async with get_session() as db:
        coins = (await db.execute(select(Coin).where(Coin.enabled == True))).scalars().all()
    symbols = []
    for coin in coins:
        loop.set_auto(coin.symbol, tf, enable)
        symbols.append(coin.symbol)
    state = "ВКЛ" if enable else "ВЫКЛ"
    logger.info("auto-all %s: %s для %d монет", tf, state, len(symbols))
    return {"tf": tf, "enable": enable, "symbols": symbols, "count": len(symbols)}


@router.get("/analytics")
async def analytics(request: Request) -> HTMLResponse:
    """Страница аналитики: таблица сделок, метрики."""
    from app.main import TEMPLATES

    async with get_session() as db:
        portfolio = get_portfolio()
        session = await portfolio.get_or_create_session(db, await get_active_mode())
        report = await portfolio.compute_analytics(db, session)
        trades = await portfolio.get_recent_trades(db, session, limit=50)
        equity_curve = await portfolio.get_equity_curve(db, session, limit=200)

    # Извлекаем данные в рамках активной сессии (DetachedInstance fix).
    trade_rows = []
    for t in trades:
        coin_sym = "—"
        try:
            if t.position and t.position.coin:
                coin_sym = t.position.coin.symbol
        except Exception:
            pass
        trade_rows.append({
            "executed_at": t.executed_at,
            "coin_symbol": coin_sym,
            "side": t.side,
            "qty": t.qty,
            "price": t.price,
            "fee": t.fee,
            "slippage": t.slippage,
            "pnl": t.pnl,
            "reason": t.reason,
        })

    eq_x = [p.timestamp.isoformat() for p in equity_curve]
    eq_y = [p.equity for p in equity_curve]

    return TEMPLATES.TemplateResponse("analytics.html", {
        "request": request,
        "session": session,
        "report": report,
        "trades": trade_rows,
        "eq_x": eq_x,
        "eq_y": eq_y,
    })


@router.get("/settings")
async def settings_page(request: Request) -> HTMLResponse:
    """Настройки стратегии + кошельки live-режима."""
    from app.main import TEMPLATES
    from core.strategy_config import get_thresholds_for_ui, THRESHOLD_KEYS
    from core.wallet_service import list_wallets_public

    live = await get_thresholds_for_ui()
    mode = await get_active_mode()
    wallets = await list_wallets_public()
    wallet_error = request.query_params.get("wallet_error", "")
    saved = request.query_params.get("saved", "")
    clamped_raw = request.query_params.get("clamped", "")
    # Парсим clamped как "key:entered→saved,key2:..." для показа в баннере.
    clamped: dict[str, tuple[float, float]] = {}
    if clamped_raw:
        for part in clamped_raw.split(","):
            if ":" in part and "→" in part:
                k, rest = part.split(":", 1)
                ent, sav = rest.split("→", 1)
                try:
                    clamped[k] = (float(ent), float(sav))
                except ValueError:
                    pass
    return TEMPLATES.TemplateResponse("settings.html", {
        "request": request,
        "settings": settings,
        "thresholds": live,
        "threshold_meta": THRESHOLD_KEYS,
        "active_mode": mode,
        "wallets": wallets,
        "wallet_error": wallet_error,
        "saved": saved,
        "clamped": clamped,
        "threshold_labels": {k: m["label"] for k, m in THRESHOLD_KEYS.items()},
    })


@router.post("/settings")
async def save_settings(request: Request):
    """Сохранить редактируемые пороги. Перенаправляет на GET с флеш-баннером."""
    from urllib.parse import urlencode
    from fastapi.responses import RedirectResponse
    from core.strategy_config import save_thresholds

    form = await request.form()
    updates = {}
    for key in form:
        try:
            updates[key] = float(form[key])
        except (ValueError, TypeError):
            pass
    _saved_display, clamped = await save_thresholds(updates, form_is_percent=True)
    # Формируем query: saved=1 + clamped=... для флеш-баннера.
    params = {"saved": "1"}
    if clamped:
        parts = [
            f"{k}:{ent:.4g}→{sav:.4g}" for k, (ent, sav) in clamped.items()
        ]
        params["clamped"] = ",".join(parts)
    return RedirectResponse(url=f"/settings?{urlencode(params)}", status_code=303)


# ─── Переключение режима paper/live ─────────────────────────


@router.get("/api/mode")
async def get_mode_api() -> dict:
    """Текущий активный режим (для JS/polling)."""
    from db.models import SessionMode
    mode = await get_active_mode()
    return {"mode": mode.value}


@router.post("/api/mode")
async def set_mode_api(request: Request) -> dict:
    """Переключить активный режим paper ↔ live."""
    from db.models import SessionMode
    from core.mode_manager import set_active_mode as set_mode
    form = await request.form()
    new_mode = str(form.get("mode", "paper"))
    if new_mode not in ("paper", "live"):
        new_mode = "paper"
    mode_enum = SessionMode.live if new_mode == "live" else SessionMode.paper
    result = await set_mode(mode_enum)
    return {"mode": result.value}


# ─── Управление кошельками (live-режим) ─────────────────────


@router.post("/settings/wallet")
async def add_wallet(request: Request):
    """Добавить кошелёк (с валидацией ключей на бирже)."""
    from fastapi.responses import RedirectResponse
    from core.wallet_service import create_wallet, WalletValidationError

    form = await request.form()
    label = str(form.get("wallet_label", ""))
    exchange = str(form.get("wallet_exchange", "binance"))
    api_key = str(form.get("wallet_api_key", ""))
    api_secret = str(form.get("wallet_api_secret", ""))

    try:
        async with get_session() as db:
            await create_wallet(db, label, exchange, api_key, api_secret)
    except WalletValidationError as e:
        # Ошибка валидации — передаём через query param.
        return RedirectResponse(
            url=f"/settings?wallet_error={str(e)}", status_code=303,
        )
    except Exception as e:
        return RedirectResponse(
            url=f"/settings?wallet_error={str(e)}", status_code=303,
        )
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/wallet/{wallet_id}/default")
async def set_wallet_default(request: Request, wallet_id: int):
    """Сделать кошелёк активным."""
    from fastapi.responses import RedirectResponse
    from core.wallet_service import set_default
    async with get_session() as db:
        await set_default(db, wallet_id)
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/wallet/{wallet_id}/delete")
async def delete_wallet(request: Request, wallet_id: int):
    """Удалить кошелёк."""
    from fastapi.responses import RedirectResponse
    from core.wallet_service import delete_wallet
    async with get_session() as db:
        await delete_wallet(db, wallet_id)
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/live-auto-toggle")
async def toggle_live_auto(request: Request):
    """Включить/выключить авто-торговлю реальными деньгами."""
    from fastapi.responses import RedirectResponse
    from config import settings
    form = await request.form()
    val = str(form.get("live_auto", "0"))
    settings.live_auto_enabled = val == "1"
    return RedirectResponse(url="/settings", status_code=303)


# ─── Вспомогательные для Plotly ──────────────────────────────


def _ohlc_for_plotly(candles) -> list:
    """Формат OHLCV для Plotly candlestick chart."""
    return [
        {
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume,
        }
        for c in candles
    ]


def _timestamps_for_plotly(candles) -> list:
    """Милисекунды epoch → datetime string для Plotly оси.

    ВАЖНО: суффикс 'Z' обязателен — без него Plotly трактует строку как
    локальное время браузера, а прогноз (через Date.toISOString()) отдаётся
    в UTC. Несовпадение поясов → линия прогноза «уезжает» на N часов назад
    относительно свечей (N = смещение TZ пользователя).
    """
    return [
        datetime.utcfromtimestamp(c.timestamp / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
        for c in candles
    ]


def _accuracy_for_plotly(candles, accuracy) -> list:
    """Маркеры точности Kronos, привязанные к timestamps графика.

    Возвращает список [{timestamp, correct, predicted_up, actual_up}, ...]
    только для тех свечей, что есть в accuracy (последние 16).
    """
    if not accuracy:
        return []
    # Привязка по timestamp: точность считается на candles[-16:].
    acc_by_ts = {r["timestamp"]: r for r in accuracy}
    result = []
    for c in candles:
        if c.timestamp in acc_by_ts:
            r = acc_by_ts[c.timestamp]
            result.append({
                "timestamp": datetime.utcfromtimestamp(c.timestamp / 1000).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "correct": r["correct"],
                "predicted_up": r["predicted_up"],
                "actual_up": r["actual_up"],
            })
    return result


def _accuracy_summary(accuracy) -> dict:
    """Сводка точности для карточки: {correct, total, pct}."""
    if not accuracy:
        return {"correct": 0, "total": 0, "pct": 0.0}
    correct = sum(1 for r in accuracy if r["correct"])
    total = len(accuracy)
    return {
        "correct": correct,
        "total": total,
        "pct": round(correct / total * 100, 1) if total else 0.0,
    }


def _drawdown_series(equity: list[float]) -> list[float]:
    """Серия просадок для графика (в %)."""
    dd = []
    peak = equity[0]
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd.append((peak - v) / peak * 100)
        else:
            dd.append(0.0)
    return dd
