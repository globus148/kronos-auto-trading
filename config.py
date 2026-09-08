"""Конфигурация приложения автоторговли.

Единый источник настроек. Загружается из переменных окружения (.env),
см. .env.example. Доступ: `from config import settings`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Все настройки приложения. Значения по умолчанию совпадают с планом.md."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Приложение ───
    app_env: Literal["development", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    # ─── Авторизация (для удалённого доступа через Cloudflare Tunnel) ───
    # Задайте пароль здесь. Если пуст — авторизация отключена (только для localhost).
    app_password: str = ""

    # ─── Live-режим (реальные деньги) ───
    # Ключ для шифрования api_secret кошельков (Fernet). Если пуст —
    # выводится предупреждение и ключ производным образом выводится из
    # app_password (только для прототипа; в продакшене задайте явный ключ).
    wallet_encryption_key: str = ""
    # Глобальный предохранитель авто-торговли реальными деньгами. По
    # умолчанию выключен — даже если включить авто на монете в live-режиме,
    # реальные ордера не ставятся, пока это не разрешено здесь. Меняется в
    # настройках осознанно.
    live_auto_enabled: bool = False

    # ─── БД ───
    database_url: str = "sqlite+aiosqlite:///./data/trading.db"

    # ─── Биржа ───
    exchange: str = "binance"
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # ─── Тестовый режим ───
    paper_initial_balance: float = 100.0

    # ─── Kronos ───
    kronos_model: str = "NeoQuasar/Kronos-base"
    kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_device: Literal["cuda", "cpu"] = "cuda"
    kronos_top_p: float = 0.95
    kronos_temperature: float = 1.2
    kronos_sample_count: int = 3
    kronos_pred_len: int = 8

    # ─── Новости (сентимент через LLM + поиск) ───
    # Провайдер: "groq" (1000/день, дефолт), "openrouter" (50/день), "gemini" (legacy).
    news_provider: str = "groq"
    # Groq: https://console.groq.com/keys (бесплатно, 1000 req/день).
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    # OpenRouter: https://openrouter.ai/keys (бесплатно, 50 req/день).
    openrouter_api_key: str = ""
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    # Tavily Search: https://tavily.com (1000 запросов/мес бесплатно).
    tavily_api_key: str = ""
    # HTTP-прокси для LLM/Tavily (обход geo-block). Xray: http://127.0.0.1:10809.
    news_proxy: str = ""
    # Gemini (legacy — оставляем, но не используется при provider!=gemini).
    gemini_api_key: str = ""
    news_model: str = "gemini-flash-lite-latest"
    news_enabled: bool = True
    news_cache_ttl: int = 1800
    news_lookback_days: int = 3

    # ─── Управление данными (план, раздел 6.1) ───
    data_kronos_context: int = 512
    data_indicator_warmup: int = 200
    data_live_buffer: int = 100
    data_backtest_window: int = 2000
    data_retention_multiplier: float = 1.1

    # ─── Стратегия (план, раздел 4) ───
    commission_rate: float = 0.001          # 0.1%
    slippage_rate: float = 0.0005           # 0.05%
    margin_of_safety: float = 0.001         # запас для порога безубыточности
    risk_fraction: float = 0.01             # 1% на сделку
    entry_threshold: float = 0.55           # порог S_entry
    atr_k_stop: float = 1.5
    atr_k_take: float = 3.0
    max_position_pct: float = 0.30
    max_hold_periods: int = 48
    # Фильтры входа (план 4.3) — настраиваемые для тестирования.
    strategy_min_rr: float = 1.5            # минимальный Risk/Reward
    strategy_min_adx: float = 15.0          # минимальный ADX (сила тренда)

    # ─── Монеты и таймфреймы по умолчанию ───
    default_coins: list[str] = Field(
        default_factory=lambda: [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "POL/USDT",
            "TON/USDT", "TRX/USDT", "LTC/USDT", "NEAR/USDT", "APT/USDT",
            "ARB/USDT", "OP/USDT", "ATOM/USDT", "DOT/USDT", "INJ/USDT",
        ]
    )
    default_timeframes: list[str] = Field(default_factory=lambda: ["1h", "4h"])

    # ─── Производные величины (план, раздел 4.1) ───

    @computed_field  # type: ignore[misc]
    @property
    def round_trip_cost(self) -> float:
        """Стоимость round-trip сделки: вход + выход (план 4.1)."""
        return 2.0 * (self.commission_rate + self.slippage_rate)

    @computed_field  # type: ignore[misc]
    @property
    def min_expected_move(self) -> float:
        """Минимальное прогнозируемое движение, при котором сделка ещё имеет смысл.

        Гарантия «выхода в плюс даже с комиссией»: сделка открывается только если
        ожидаемое движение >= этого порога.
        """
        return self.round_trip_cost + self.margin_of_safety

    @computed_field  # type: ignore[misc]
    @property
    def required_history(self) -> int:
        """Сколько свечей нужно на старте для каждой пары/ТФ (план 6.1)."""
        return max(
            self.data_kronos_context + self.data_live_buffer,
            self.data_indicator_warmup + self.data_live_buffer,
            self.data_backtest_window,
        )

    @computed_field  # type: ignore[misc]
    @property
    def retention_size(self) -> int:
        """Размер скользящего окна хранения свечей (план 6.1)."""
        return int(self.required_history * self.data_retention_multiplier)


@lru_cache
def get_settings() -> Settings:
    """Синглтон настроек (кешируется на весь процесс)."""
    return Settings()


# Глобальный экземпляр для удобного импорта: `from config import settings`
settings = get_settings()
