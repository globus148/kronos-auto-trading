"""RiskManager — управление размером позиции и стоп-лоссами (план 4.5).

Реализует Fixed-Fractional подход:
    - Риск на сделку = risk_fraction × equity (по умолчанию 1%)
    - Стоп-лосс = k_sl × ATR (по умолчанию 1.5)
    - Размер позиции = risk_amount / stop_distance
    - Максимум max_position_pct от equity в одной позиции

Все расчёты возвращаются как PositionSizing — готовые числа для PortfolioEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config import settings
from core.strategy_config import get_snapshot


@dataclass
class PositionSizing:
    """Расчёт размера позиции и уровней стопов.

    Возвращает всё что нужно PortfolioEngine для открытия позиции.
    """
    qty: float               # количество монет для покупки
    position_value: float   # USDT в позиции
    stop_price: float        # цена стоп-лосса
    target_price: float      # цена тейк-профита
    risk_amount: float       # $ риска (magnitude)
    risk_pct: float          # risk / equity
    stop_pct: float          # стоп в % от входа
    rr_ratio: float          # reward/risk (должен быть ≥ 2)
    break_even_price: float  # цена безубыточности (с учётом комиссии)

    def __post_init__(self) -> None:
        # Валидация.
        if self.qty <= 0:
            raise ValueError(f"qty={self.qty} должна быть > 0")
        if self.rr_ratio < 1.0:
            raise ValueError(f"rr_ratio={self.rr_ratio:.2f} < 1 — невыгодная сделка")


class RiskManager:
    """Расчёт размера позиции по Fixed-Fractional + ATR (план 4.5)."""

    def __init__(
        self,
        risk_fraction: float | None = None,
        atr_k_stop: float | None = None,
        atr_k_take: float | None = None,
        max_position_pct: float | None = None,
        commission_rate: float | None = None,
        slippage_rate: float | None = None,
    ) -> None:
        # Явные overrides (для тестов). None → читать live-снапшот в момент расчёта.
        self._overrides: dict[str, float | None] = {
            "risk_fraction": risk_fraction,
            "atr_k_stop": atr_k_stop,
            "atr_k_take": atr_k_take,
            "max_position_pct": max_position_pct,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
        }

    # Свойства с lazy-чтением live-порогов (если override не задан).
    @property
    def risk_fraction(self) -> float:
        return self._ov("risk_fraction")

    @property
    def atr_k_stop(self) -> float:
        return self._ov("atr_k_stop")

    @property
    def atr_k_take(self) -> float:
        return self._ov("atr_k_take")

    @property
    def max_position_pct(self) -> float:
        return self._ov("max_position_pct")

    @property
    def commission_rate(self) -> float:
        return self._ov("commission_rate")

    @property
    def slippage_rate(self) -> float:
        return self._ov("slippage_rate")

    def _ov(self, key: str) -> float:
        """Override (если задан) иначе live-снапшот иначе дефолт config.settings."""
        val = self._overrides.get(key)
        if val is not None:
            return float(val)
        return get_snapshot().get(key, float(getattr(settings, key)))

    def calculate(
        self,
        entry_price: float,
        atr: float,
        equity: float,
        cash: float,
        target_from_prediction: float | None = None,
    ) -> PositionSizing:
        """Рассчитать размер позиции для LONG входа.

        Args:
            entry_price: цена входа (текущая close).
            atr: текущее значение ATR (волатильность).
            equity: текущий капитал (cash + positions_value).
            cash: свободный USDT.
            target_from_prediction: целевая цена из Kronos (опционально).

        Returns:
            PositionSizing с рассчитанными параметрами.
        """
        # 1. Риск в долларах.
        risk_amount = self.risk_fraction * equity

        # 2. Стоп-лосс в цене и в %.
        stop_distance = self.atr_k_stop * atr
        stop_price = entry_price - stop_distance
        stop_pct = stop_distance / entry_price

        # Защита: стоп не должен быть глубже 10% (аномальная волатильность).
        if stop_pct > 0.10:
            stop_pct = 0.10
            stop_distance = entry_price * stop_pct
            stop_price = entry_price - stop_distance

        # 3. Тейк-профит: лучший из прогноза или ATR-таргета.
        # Если прогноз близко к entry → rr_ratio < 1 и сделка падает с ValueError.
        # Берём максимум: прогноз (если выгоднее) ИЛИ ATR-таргет (гарантирует R/R).
        target_distance_atr = self.atr_k_take * atr
        if target_from_prediction and target_from_prediction > entry_price:
            pred_target = entry_price + (target_from_prediction - entry_price) * 0.9
            # Прогноз выгоднее ATR — берём его, иначе fallback на ATR.
            if (pred_target - entry_price) >= target_distance_atr:
                target_price = pred_target
            else:
                target_price = entry_price + target_distance_atr
        else:
            target_price = entry_price + target_distance_atr

        # 4. Reward/Risk.
        potential_profit = target_price - entry_price
        rr_ratio = potential_profit / stop_distance if stop_distance > 0 else 0.0

        # 5. Размер позиции из Fixed-Fractional.
        if stop_pct > 0:
            position_value = risk_amount / stop_pct
        else:
            position_value = 0.0

        # 6. Ограничение: max_position_pct от equity и доступный cash.
        max_value = min(
            self.max_position_pct * equity,
            cash,
        )
        position_value = min(position_value, max_value)

        # 7. Количество монет.
        qty = position_value / entry_price if entry_price > 0 else 0.0

        # 8. Цена безубыточности (round_trip_cost).
        total_cost_pct = 2.0 * (self.commission_rate + self.slippage_rate)
        break_even_price = entry_price * (1.0 + total_cost_pct)

        return PositionSizing(
            qty=qty,
            position_value=position_value,
            stop_price=stop_price,
            target_price=target_price,
            risk_amount=risk_amount,
            risk_pct=risk_amount / equity if equity > 0 else 0.0,
            stop_pct=stop_pct,
            rr_ratio=rr_ratio,
            break_even_price=break_even_price,
        )

    def calculate_short(
        self,
        entry_price: float,
        atr: float,
        equity: float,
        cash: float,
        target_from_prediction: float | None = None,
    ) -> PositionSizing:
        """Рассчитать размер SHORT-позиции (зеркально к calculate).

        Для short стоп выше входа, тейк ниже. Прибыль = падение цены.

        Args:
            entry_price: цена входа (текущая close).
            atr: текущее значение ATR.
            equity: текущий капитал.
            cash: свободный USDT.
            target_from_prediction: целевая (минимальная) цена из Kronos
                (pred_min_low). Должна быть ниже entry_price.
        """
        # 1. Риск в долларах.
        risk_amount = self.risk_fraction * equity

        # 2. Стоп-лосс ВЫШЕ входа (для short убыток при росте цены).
        stop_distance = self.atr_k_stop * atr
        stop_price = entry_price + stop_distance
        stop_pct = stop_distance / entry_price

        # Защита: стоп не глубже 10% от входа.
        if stop_pct > 0.10:
            stop_pct = 0.10
            stop_distance = entry_price * stop_pct
            stop_price = entry_price + stop_distance

        # 3. Тейк-профит: лучший из прогноза или ATR-таргета (зеркально к LONG).
        target_distance_atr = self.atr_k_take * atr
        if target_from_prediction and target_from_prediction < entry_price:
            pred_target = entry_price - (entry_price - target_from_prediction) * 0.9
            # Прогноз выгоднее ATR — берём его, иначе fallback на ATR.
            if (entry_price - pred_target) >= target_distance_atr:
                target_price = pred_target
            else:
                target_price = entry_price - target_distance_atr
        else:
            target_price = entry_price - target_distance_atr

        # 4. Reward/Risk (для short potential_profit = падение).
        potential_profit = entry_price - target_price
        rr_ratio = potential_profit / stop_distance if stop_distance > 0 else 0.0

        # 5. Размер позиции из Fixed-Fractional.
        if stop_pct > 0:
            position_value = risk_amount / stop_pct
        else:
            position_value = 0.0

        # 6. Ограничение по max_position_pct и cash (обеспечение short = position_value).
        max_value = min(
            self.max_position_pct * equity,
            cash,
        )
        position_value = min(position_value, max_value)

        # 7. Количество монет.
        qty = position_value / entry_price if entry_price > 0 else 0.0

        # 8. Цена безубыточности для short (ниже входа на round_trip_cost).
        total_cost_pct = 2.0 * (self.commission_rate + self.slippage_rate)
        break_even_price = entry_price * (1.0 - total_cost_pct)

        return PositionSizing(
            qty=qty,
            position_value=position_value,
            stop_price=stop_price,
            target_price=target_price,
            risk_amount=risk_amount,
            risk_pct=risk_amount / equity if equity > 0 else 0.0,
            stop_pct=stop_pct,
            rr_ratio=rr_ratio,
            break_even_price=break_even_price,
        )
