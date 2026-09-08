"""StrategyEngine — аналитическая модель: сигналы entry/exit (план 4.3–4.6).

Ядро стратегии: ансамбль 5 суб-сигналов → композитный S_entry/S_exit.
Фильтры безубыточности: комиссионный порог, risk/reward, expected_pnl.

Зависит от:
    - core.indicators.IndicatorSnapshot (классические индикаторы)
    - (опционально) core.prediction_service (Kronos прогноз)
    - core.risk_manager.RiskManager (sizing)
    - config.settings (пороги)

Когда Kronos ещё не подключён (этап 4), f_kronos = 0 — стратегия
работает чисто на индикаторах с компенсирующими весами.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from config import settings
from core.indicators import IndicatorSnapshot
from core.risk_manager import PositionSizing, RiskManager
from core.strategy_config import get_snapshot

logger = logging.getLogger("trading.strategy")


def _thr(key: str) -> float:
    """Редактируемый порог из снапшота (меняется через /settings)."""
    return get_snapshot().get(key, float(getattr(settings, key)))

# ─── Типы решений ───────────────────────────────────────────


@dataclass
class SignalResult:
    """Результат анализа на текущем баре."""
    s_entry: float               # композитный сигнал входа LONG ∈ [0, 1]
    s_entry_short: float         # композитный сигнал входа SHORT ∈ [0, 1]
    s_exit: float                # композитный сигнал выхода ∈ [0, 1]
    should_enter: bool           # финальное решение: входить ли
    exit_reason: str | None      # причина выхода (если есть)
    sub_signals: dict[str, float]  # отдельные суб-сигналы LONG для UI
    sub_signals_short: dict[str, float] = field(default_factory=dict)  # SHORT для UI


class StrategyEngine:
    """Расчёт композитных сигналов entry/exit (план 4.3–4.6)."""

    def __init__(self, risk_mgr: RiskManager | None = None) -> None:
        self.risk_mgr = risk_mgr or RiskManager()

    # ─── 4.3: Композитный сигнал входа ───────────────────────

    def compute_signal(
        self,
        snap: IndicatorSnapshot,
        kronos_pred_return: float | None = None,
        kronos_pred_slope: float | None = None,
        kronos_pred_max: float | None = None,
        kronos_pred_min: float | None = None,
        news_sentiment: float | None = None,
    ) -> SignalResult:
        """Посчитать S_entry (long), S_entry_short и S_exit, принять решение.

        Args:
            snap: снапшот индикаторов на текущем баре.
            kronos_pred_return: ожидаемая доходность от Kronos (опционально).
            kronos_pred_slope: наклон прогноза (опционально).
            kronos_pred_max: ожидаемый max high от Kronos (опционально).
            kronos_pred_min: ожидаемый min low от Kronos (опционально, для short-target).
            news_sentiment: настройка новостей ∈ [-1, +1] (опц., None = нет данных).
        """
        subs = self._sub_signals(
            snap, kronos_pred_return, kronos_pred_slope, news_sentiment,
        )
        subs_short = self._sub_signals_short(
            snap, kronos_pred_return, kronos_pred_slope, news_sentiment,
        )
        s_entry = self._composite_signal(subs)
        s_entry_short = self._composite_signal(subs_short)  # веса те же, значения зеркальные
        s_exit = self._composite_exit_signal(snap, kronos_pred_return, kronos_pred_slope)

        return SignalResult(
            s_entry=s_entry,
            s_entry_short=s_entry_short,
            s_exit=s_exit,
            should_enter=False,  # заполняется через evaluate_entry()
            exit_reason=None,
            sub_signals=subs,
            sub_signals_short=subs_short,
        )

    def evaluate_entry(
        self,
        snap: IndicatorSnapshot,
        s_entry: float,
        sub_signals: dict[str, float],
        equity: float,
        cash: float,
        kronos_pred_return: float | None = None,
        kronos_pred_max: float | None = None,
    ) -> tuple[bool, PositionSizing | None, str]:
        """Полная оценка: стоит ли входить, и если да — какой sizing.

        Применяет все фильтры плана 4.3 (AND-условия).
        Returns (should_enter, position_sizing_or_None, reject_reason).
        """
        # 1. Порог S_entry.
        if s_entry < _thr("entry_threshold"):
            reason = f"s_entry={s_entry:.3f} < порог {_thr('entry_threshold'):.2f}"
            logger.info("evaluate_entry: отказ — %s", reason)
            return False, None, reason

        # 2. Комиссионный фильтр (план 4.1).
        if kronos_pred_return is not None:
            if kronos_pred_return < settings.min_expected_move:
                reason = f"pred_return={kronos_pred_return:.4f} < min {settings.min_expected_move:.4f}"
                logger.info("evaluate_entry: отказ — %s", reason)
                return False, None, reason
        # Без Kronos больше НЕ требуем завышенного s_entry>=0.80 — используем
        # тот же редактируемый entry_threshold. Иначе вход почти невозможен.

        # 3. ADX — есть тренд (план 4.3).
        if snap.adx < _thr("strategy_min_adx"):
            reason = f"ADX={snap.adx:.1f} < порог {_thr('strategy_min_adx'):.1f}"
            logger.info("evaluate_entry: отказ — %s", reason)
            return False, None, reason

        # 4. Расчёт sizing.
        try:
            sizing = self.risk_mgr.calculate(
                entry_price=snap.close,
                atr=snap.atr,
                equity=equity,
                cash=cash,
                target_from_prediction=kronos_pred_max,
            )
        except (ValueError, ZeroDivisionError):
            reason = "ошибка расчёта размера позиции"
            logger.info("evaluate_entry: отказ — %s", reason)
            return False, None, reason

        # 5. Risk/Reward ≥ порога (план 4.3).
        # Tolerance 0.01: защита от float-precision (atr_k_take/atr_k_stop=2.0
        # даёт 1.9999... при не-круглом ATR и сделка ошибочно отбрасывается).
        if sizing.rr_ratio < _thr("strategy_min_rr") - 0.01:
            reason = f"R/R={sizing.rr_ratio:.2f} < порог {_thr('strategy_min_rr'):.2f}"
            logger.info("evaluate_entry: отказ — %s", reason)
            return False, None, reason

        # 6. Фильтр expected_pnl > 0 (план 4.6).
        # Порог — 2% от position_value (а не фиксированные -$0.05).
        # При $100 балансе position_value ≈ $30 → порог -$0.60,
        # раньше было -$0.05 и блокировало почти все сделки.
        p_win = self.estimate_win_rate(s_entry, kronos_pred_return)
        best_case_pnl = sizing.qty * (sizing.target_price - snap.close)
        round_trip_cost_val = (
            sizing.qty * snap.close
            * 2.0 * (settings.commission_rate + settings.slippage_rate)
        )
        best_case_pnl -= round_trip_cost_val
        worst_case_pnl = -sizing.risk_amount
        expected_pnl = p_win * best_case_pnl + (1 - p_win) * worst_case_pnl
        # Порог в абсолютных$: 2% от размера позиции.
        ep_threshold = -0.02 * sizing.position_value
        if expected_pnl <= ep_threshold:
            reason = f"expected_pnl={expected_pnl:.4f} < {ep_threshold:.4f} (p_win={p_win:.2f})"
            logger.info("evaluate_entry: отказ — %s", reason)
            return False, None, reason

        return True, sizing, ""

    # ─── 4.3short: Оценка входа в SHORT ───────────────────────

    def evaluate_short_entry(
        self,
        snap: IndicatorSnapshot,
        s_entry_short: float,
        sub_signals_short: dict[str, float],
        equity: float,
        cash: float,
        kronos_pred_return: float | None = None,
        kronos_pred_min: float | None = None,
    ) -> tuple[bool, PositionSizing | None, str]:
        """Полная оценка входа в SHORT. Зеркально к evaluate_entry.

        Returns (should_short, position_sizing_or_None, reject_reason).
        """
        # 1. Порог S_entry_short.
        if s_entry_short < _thr("entry_threshold"):
            reason = f"s_entry={s_entry_short:.3f} < порог {_thr('entry_threshold'):.2f}"
            logger.info("evaluate_short_entry: отказ — %s", reason)
            return False, None, reason

        # 2. Комиссионный фильтр: нужен прогноз на падение.
        if kronos_pred_return is not None:
            if kronos_pred_return > -settings.min_expected_move:
                reason = f"pred_return={kronos_pred_return:.4f} > -min {-settings.min_expected_move:.4f}"
                logger.info("evaluate_short_entry: отказ — %s", reason)
                return False, None, reason
        # Без Kronos используем тот же entry_threshold (без завышения 0.80).

        # 3. ADX — есть тренд.
        if snap.adx < _thr("strategy_min_adx"):
            reason = f"ADX={snap.adx:.1f} < порог {_thr('strategy_min_adx'):.1f}"
            logger.info("evaluate_short_entry: отказ — %s", reason)
            return False, None, reason

        # 4. Sizing short.
        try:
            sizing = self.risk_mgr.calculate_short(
                entry_price=snap.close,
                atr=snap.atr,
                equity=equity,
                cash=cash,
                target_from_prediction=kronos_pred_min,
            )
        except (ValueError, ZeroDivisionError):
            reason = "ошибка расчёта размера позиции"
            logger.info("evaluate_short_entry: отказ — %s", reason)
            return False, None, reason

        # 5. Risk/Reward ≥ порога.
        # Tolerance 0.01: защита от float-precision (см. evaluate_entry).
        if sizing.rr_ratio < _thr("strategy_min_rr") - 0.01:
            reason = f"R/R={sizing.rr_ratio:.2f} < порог {_thr('strategy_min_rr'):.2f}"
            logger.info("evaluate_short_entry: отказ — %s", reason)
            return False, None, reason

        # 6. Фильтр expected_pnl > 0.
        # Порог — 2% от position_value (а не фиксированные -$0.05).
        p_win = self.estimate_win_rate(s_entry_short, kronos_pred_return)
        # Для short: best_case = прибыль при падении до target.
        best_case_pnl = sizing.qty * (snap.close - sizing.target_price)
        round_trip_cost_val = (
            sizing.qty * snap.close
            * 2.0 * (settings.commission_rate + settings.slippage_rate)
        )
        best_case_pnl -= round_trip_cost_val
        worst_case_pnl = -sizing.risk_amount
        expected_pnl = p_win * best_case_pnl + (1 - p_win) * worst_case_pnl
        ep_threshold = -0.02 * sizing.position_value
        if expected_pnl <= ep_threshold:
            reason = f"expected_pnl={expected_pnl:.4f} < {ep_threshold:.4f} (p_win={p_win:.2f})"
            logger.info("evaluate_short_entry: отказ — %s", reason)
            return False, None, reason

        return True, sizing, ""

    # ─── 4.4: Проверка выхода по открытой позиции ───────────

    def evaluate_exit(
        self,
        snap: IndicatorSnapshot,
        position_price: float,
        stop_price: float,
        target_price: float,
        bars_held: int,
        trailing_stop: float | None = None,
        highest_since_entry: float | None = None,
        kronos_pred_return: float | None = None,
        kronos_pred_slope: float | None = None,
        side: Literal["long", "short"] = "long",
    ) -> str | None:
        """Проверить exit-условия. Возвращает reason или None (держим).

        Приоритет (первый сработавший) — план 4.4:
        1. take_profit
        2. trailing_stop
        3. stop_loss
        4. signal_reversal
        5. time_stop

        side: для long профит при росте (close≥target), стоп при падении (close≤stop).
              Для short инвертируется: профит при падении (close≤target), стоп при росте.
        """
        is_short = side == "short"

        # 1. Take-profit.
        if not is_short and snap.close >= target_price:
            return "take_profit"
        if is_short and snap.close <= target_price:
            return "take_profit"

        # 2. Trailing stop (если активирован).
        # Для long trailing НИЖЕ цены (стоп при падении к нему).
        # Для short trailing ВЫШЕ цены (стоп при росте к нему).
        if trailing_stop is not None:
            if not is_short and snap.close <= trailing_stop:
                return "trailing_stop"
            if is_short and snap.close >= trailing_stop:
                return "trailing_stop"

        # 3. ATR stop-loss (жёсткий).
        if not is_short and snap.close <= stop_price:
            return "stop_loss"
        if is_short and snap.close >= stop_price:
            return "stop_loss"

        # 4. Signal reversal (против направления позиции).
        if kronos_pred_return is not None and kronos_pred_slope is not None:
            if not is_short:
                # Разворот ВНИЗ против long.
                if (kronos_pred_slope < 0
                        and kronos_pred_return < -settings.min_expected_move):
                    return "signal_reversal"
            else:
                # Разворот ВВЕРХ против short.
                if (kronos_pred_slope > 0
                        and kronos_pred_return > settings.min_expected_move):
                    return "signal_reversal"
        else:
            # Без Kronos: для long — RSI перекуплен + MACD вниз.
            #             Для short — RSI перепродан + MACD вверх.
            if not is_short:
                if snap.rsi > 75 and snap.macd_hist < 0:
                    return "signal_reversal"
            else:
                if snap.rsi < 25 and snap.macd_hist > 0:
                    return "signal_reversal"

        # 5. Time stop.
        if bars_held >= _thr("max_hold_periods"):
            return "time_stop"

        return None

    def update_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        current_trailing: float | None,
        atr: float,
        highest_since_entry: float | None = None,
        activation_rr: float = 1.0,
    ) -> float | None:
        """Обновить trailing stop.

        Активируется когда цена достигла 1R прибыли. После активации
        следует за ценой на дистанции k * ATR.
        """
        risk_mgr = self.risk_mgr
        risk = risk_mgr.risk_fraction * 100  # proxy for 1R (rough)
        activation_price = entry_price + risk

        if highest_since_entry is None:
            highest_since_entry = current_price

        if current_price >= activation_price:
            # Trailing активирован.
            new_trail = highest_since_entry - risk_mgr.atr_k_stop * atr
            if current_trailing is None or new_trail > current_trailing:
                return new_trail

        return current_trailing

    # ─── Внутренние: суб-сигналы ─────────────────────────────

    def _sub_signals(
        self,
        snap: IndicatorSnapshot,
        pred_return: float | None,
        pred_slope: float | None,
        news_sentiment: float | None = None,
    ) -> dict[str, float]:
        """Посчитать отдельные суб-сигналы ∈ [0, 1]."""
        signals: dict[str, float] = {}

        # f_trend: EMA20 > EMA50 и обе растут (наклон > 0).
        ema_cross = 1.0 if snap.ema20 > snap.ema50 else 0.0
        # Наклон EMA20 ≈ (ema20 - ema50) / ema50, нормализуем в [0,1].
        ema_spread = (snap.ema20 - snap.ema50) / snap.ema50 if snap.ema50 > 0 else 0
        signals["f_trend"] = max(0.0, min(1.0, ema_cross * 0.6 + ema_spread * 20))

        # f_kronos: прогноз Kronos положительный и растущий.
        # Шкала: +0.5% → ~0.45, +1% → ~0.6, +3% → ~0.85, +5% → ~0.95.
        # Сильный прогноз должен давать высокий сигнал (а не потолок 0.65).
        if pred_return is not None and pred_slope is not None:
            kronos_ok = pred_return >= settings.min_expected_move
            slope_ok = pred_slope > 0
            # Сила: 0 при 0%, 1 при +3%, насыщение выше.
            strength = min(1.0, max(0.0, pred_return) / 0.03)
            if kronos_ok:
                # База 0.4..0.95 по силе прогноза.
                kronos_val = 0.40 + 0.55 * strength
            else:
                # Прогноз положительный, но слабый (< min_expected_move).
                kronos_val = 0.40 * strength
            if slope_ok:
                kronos_val = min(1.0, kronos_val * 1.1)
            signals["f_kronos"] = max(0.0, min(1.0, kronos_val))
        else:
            signals["f_kronos"] = 0.0  # Kronos ещё не подключён.

        # f_momentum: MACD histogram > 0 (бычий импульс).
        # Раньше: при hist<0 строго 0 — это обнуляло целый вес 0.20.
        # Теперь: даём ненулевой сигнал даже при негативе (0.35..0.50),
        # чтобы умеренный медвежий импульс (коррекция перед ростом) не
        # блокировал вход при согласии остальных сигналов тренда.
        hist_strength = min(1.0, abs(snap.macd_hist) / (snap.close * 0.005))
        if snap.macd_hist > 0:
            signals["f_momentum"] = 0.5 + 0.5 * hist_strength  # 0.5..1.0
        else:
            # Медвежий импульс — база 0.50, затухание до 0.35 при сильном негативе.
            signals["f_momentum"] = max(0.35, 0.50 - 0.15 * hist_strength)

        # f_rsi: RSI в «зоне силы» [40, 70]. Высокий при 45-65, низкий при <35 или >75.
        if 40 <= snap.rsi <= 70:
            # Пик на 55: 55 → 1.0, 40 → 0.6, 70 → 0.4
            rsi_center = 55.0
            signals["f_rsi"] = max(0.4, 1.0 - abs(snap.rsi - rsi_center) / 30.0)
        elif 35 <= snap.rsi < 40:
            # Перепроданность нарастает (но не экстремальная) — плавный подъём к зоне силы.
            # 35 → 0.3, 40 → 0.6 (стыкуется с верхней веткой).
            signals["f_rsi"] = 0.3 + (snap.rsi - 35) / 5.0 * 0.3
        elif snap.rsi < 35:
            # Перепродан — слабый сигнал входа (может быть продолжение падения).
            signals["f_rsi"] = max(0.0, snap.rsi / 70.0)
        elif 70 < snap.rsi <= 85:
            # Перекуплен — затухание.
            signals["f_rsi"] = max(0.0, 1.0 - (snap.rsi - 70) / 15.0)
        else:
            # RSI > 85 — строго 0 (опасно входить).
            signals["f_rsi"] = 0.0

        # f_pullback: цена откатилась к EMA20 — точка входа после отката.
        dist_to_ema20 = (snap.close - snap.ema20) / snap.ema20 if snap.ema20 > 0 else 0
        pullback_score = 0.0
        if -0.02 < dist_to_ema20 < 0.01 and 40 <= snap.rsi <= 55:
            # Идеальный pullback: цена чуть ниже EMA20 и RSI 40-55.
            pullback_score = 0.7 + 0.3 * (1.0 - abs(dist_to_ema20) / 0.02)
        elif dist_to_ema20 < 0 and snap.rsi < 50:
            # Мягкий откат ниже EMA20.
            pullback_score = 0.3
        elif -0.03 < dist_to_ema20 < 0.02 and snap.rsi < 60:
            # Широкая зона: цена около EMA20 и RSI не перекуплен.
            pullback_score = max(0.0, 0.15 + 0.15 * (1.0 - abs(dist_to_ema20) / 0.03))
        signals["f_pullback"] = max(0.0, min(1.0, pullback_score))

        # f_news: настроение новостей (LONG). Бычий → 1.0, медвежий → 0.0, нейтрально → 0.5.
        # None (нет ключа / сервис недоступен) = 0.5, чтобы не перекашивать композит.
        if news_sentiment is not None:
            s = max(-1.0, min(1.0, news_sentiment))
            signals["f_news"] = 0.5 + 0.5 * s
        else:
            signals["f_news"] = 0.5

        return signals

    def _sub_signals_short(
        self,
        snap: IndicatorSnapshot,
        pred_return: float | None,
        pred_slope: float | None,
        news_sentiment: float | None = None,
    ) -> dict[str, float]:
        """Посчитать суб-сигналы для SHORT (падение цены) ∈ [0, 1].

        Зеркально к _sub_signals: всё инвертировано под нисходящее движение.
        Ключи те же (f_trend, f_kronos, f_momentum, f_rsi, f_pullback),
        чтобы переиспользовать _composite_signal (веса симметричны).
        """
        signals: dict[str, float] = {}

        # f_trend: EMA20 < EMA50 и обе падают.
        ema_cross_down = 1.0 if snap.ema20 < snap.ema50 else 0.0
        ema_spread = (snap.ema50 - snap.ema20) / snap.ema50 if snap.ema50 > 0 else 0
        signals["f_trend"] = max(0.0, min(1.0, ema_cross_down * 0.6 + ema_spread * 20))

        # f_kronos: прогноз Kronos отрицательный и падающий (зеркало long).
        # Сильный прогноз падения → высокий сигнал.
        if pred_return is not None and pred_slope is not None:
            kronos_ok = pred_return <= -settings.min_expected_move
            slope_ok = pred_slope < 0
            strength = min(1.0, max(0.0, abs(pred_return)) / 0.03)  # 3% → max
            if kronos_ok:
                kronos_val = 0.40 + 0.55 * strength
            else:
                kronos_val = 0.40 * strength
            if slope_ok:
                kronos_val = min(1.0, kronos_val * 1.1)
            signals["f_kronos"] = max(0.0, min(1.0, kronos_val))
        else:
            signals["f_kronos"] = 0.0

        # f_momentum: MACD histogram < 0 (медвежий импульс для short).
        hist_strength = min(1.0, abs(snap.macd_hist) / (snap.close * 0.005))
        if snap.macd_hist < 0:
            signals["f_momentum"] = 0.5 + 0.5 * hist_strength  # 0.5..1.0
        else:
            # Бычий импульс против шорта — база 0.50, затухание до 0.35.
            signals["f_momentum"] = max(0.35, 0.50 - 0.15 * hist_strength)

        # f_rsi: для short «зона силы» [30, 60] (ослабление/падение).
        # Высокий при 35-55, низкий при перекупленности (>70).
        if 30 <= snap.rsi <= 60:
            # Пик на 45: 45 → 1.0, 60 → 0.6, 30 → 0.6.
            rsi_center = 45.0
            signals["f_rsi"] = max(0.4, 1.0 - abs(snap.rsi - rsi_center) / 30.0)
        elif 60 < snap.rsi <= 65:
            # Плавный заход с зоны силы.
            signals["f_rsi"] = 0.3 + (65 - snap.rsi) / 5.0 * 0.3
        elif snap.rsi > 65:
            # Перекуплен — хороший сигнал на шорт (затухание роста).
            signals["f_rsi"] = max(0.0, min(1.0, (snap.rsi - 65) / 20.0 + 0.4))
        elif 25 <= snap.rsi < 30:
            # Сильная перепроданность — шорт рискован (возможен отскок).
            signals["f_rsi"] = max(0.0, 0.3 - (30 - snap.rsi) / 5.0 * 0.3)
        else:
            # RSI < 25 — перепродан, шорт опасен.
            signals["f_rsi"] = 0.0

        # f_pullback: цена откатилась ВВЕРХ к EMA20 — точка входа в short после
        # контр-трендового отскока (релативно к нисходящему тренду).
        dist_to_ema20 = (snap.close - snap.ema20) / snap.ema20 if snap.ema20 > 0 else 0
        pullback_score = 0.0
        if -0.01 < dist_to_ema20 < 0.02 and 45 <= snap.rsi <= 60:
            pullback_score = 0.7 + 0.3 * (1.0 - abs(dist_to_ema20) / 0.02)
        elif dist_to_ema20 > 0 and snap.rsi > 50:
            pullback_score = 0.3
        elif -0.02 < dist_to_ema20 < 0.03 and snap.rsi > 40:
            pullback_score = max(0.0, 0.15 + 0.15 * (1.0 - abs(dist_to_ema20) / 0.03))
        signals["f_pullback"] = max(0.0, min(1.0, pullback_score))

        # f_news для SHORT — инверсия: бычьи новости = плохо для шорта.
        # Медвежий сентимент (< 0) → высокий сигнал short.
        if news_sentiment is not None:
            s = max(-1.0, min(1.0, news_sentiment))
            signals["f_news"] = 0.5 - 0.5 * s
        else:
            signals["f_news"] = 0.5

        return signals

    def _composite_signal(self, subs: dict[str, float]) -> float:
        """Взвешенная сумма суб-сигналов (план 4.3)."""
        # Если Kronos подключён — его вес 0.30; если нет — перераспределяем.
        has_kronos = subs.get("f_kronos", 0) > 0
        if has_kronos:
            weights = {
                "f_trend": 0.23, "f_kronos": 0.27, "f_momentum": 0.18,
                "f_rsi": 0.13, "f_pullback": 0.09, "f_news": 0.10,
            }
        else:
            weights = {
                "f_trend": 0.28, "f_kronos": 0.00, "f_momentum": 0.24,
                "f_rsi": 0.22, "f_pullback": 0.16, "f_news": 0.10,
            }
        total = sum(weights.get(k, 0) * v for k, v in subs.items())
        # Нормализация: если сумма весов < 1 (Kronos=0), делим на фактическую сумму.
        w_sum = sum(weights.get(k, 0) for k in subs if weights.get(k, 0) > 0)
        if w_sum > 0:
            total /= w_sum
        return max(0.0, min(1.0, total))

    def _composite_exit_signal(
        self,
        snap: IndicatorSnapshot,
        pred_return: float | None,
        pred_slope: float | None,
    ) -> float:
        """Сигнал выхода ∈ [0, 1]. Высокий = пора закрывать."""
        score = 0.0
        # RSI > 75 → сильный сигнал выхода.
        if snap.rsi > 75:
            score += 0.3 + 0.2 * min(1.0, (snap.rsi - 75) / 15)
        # MACD вниз.
        if snap.macd_hist < 0:
            score += 0.2
        # Kronos разворот.
        if pred_return is not None and pred_slope is not None:
            if pred_slope < 0:
                score += 0.3
            if pred_return < -settings.min_expected_move:
                score += 0.2
        return min(1.0, score)

    def estimate_win_rate(
        self,
        s_entry: float,
        kronos_pred_return: float | None,
    ) -> float:
        """Оценка вероятности выигрыша (p_win) для фильтра 4.6.

        Калибруется из bэктеста; пока используем эвристику по S_entry.
        Публичный метод — используется TradingLoop для live-отображения.
        """
        # Базовая p_win по скору: 0.65 → 55%, 0.85 → 70%.
        base = 0.40 + 0.35 * s_entry
        # Если Kronos согласен — прибавляем уверенности.
        if kronos_pred_return is not None and kronos_pred_return > 0.01:
            base += 0.05
        return min(0.75, max(0.35, base))
