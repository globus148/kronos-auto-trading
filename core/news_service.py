"""Новостной индикатор настроения для s_entry.

Анализирует свежие биржевые/крипто-новости по монете через LLM + поиск и
возвращает:
  - sentiment ∈ [-1.0, +1.0]  (медвежий .. бычий)
  - summary  — суммаризация новостей за последние N дней (3-5 предложений)
  - confidence ∈ [0.0, 1.0]
  - sources  — список URL источников
  - key_events — 3-5 ключевых событий

Провайдеры:
  1. OpenRouter (дефолт) — бесплатная модель Nemotron 3 Ultra (550B) +
     Tavily Search для поиска новостей.
  2. Gemini (legacy) — google-genai SDK + Google Search grounding.
  3. RSS fallback — CoinTelegraph / The Block, если поиск недоступен.

Кеш на 30 мин на символ (настраивается).
Если ключей нет — get_sentiment() возвращает None, и стратегия работает
как раньше (f_news = 0.5 нейтрально).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import aiohttp

from config import settings

logger = logging.getLogger("trading.news")


class NewsError(Exception):
    """Понятная ошибка для UI (неверный ключ / квота / сеть)."""


# ─── Системный промпт (улучшенный — просим подробный анализ) ─────────

_SYSTEM_PROMPT = (
    "Ты — финансовый аналитик криптовалют. Твоя задача: проанализировать "
    "свежие новости по конкретной монете и оценить рыночное настроение.\n\n"
    "Финальный ответ — СТРОГО валидный JSON без markdown-обёртки, в формате:\n"
    '{{\n'
    '  "sentiment": <float от -1.0 до 1.0>,\n'
    '  "summary": "<3-5 предложений на русском: ключевые события, влияние на цену, '
    'объёмы, регуляция, мнения аналитиков>",\n'
    '  "confidence": <float от 0.0 до 1.0>,\n'
    '  "key_events": ["<событие 1>", "<событие 2>", "<событие 3>", "<событие 4>"]\n'
    '}}\n\n'
    "Шкала sentiment:\n"
    "  +1.0 = сильно бычий (листинги, партнёрства, приток капитала, ETF-новости)\n"
    "  +0.3..+0.7 = умеренно позитивный\n"
    "   0.0 = нейтральный / смешанный\n"
    "  -0.3..-0.7 = умеренно негативный (регуляторное давление, хаки)\n"
    "  -1.0 = сильно медвежий (взлом, бан, иск SEC, массовый вывод)\n"
    "Если новостей мало — confidence низкий (0.2-0.4).\n"
    "key_events — 1-5 самых значимых событий из новостей (сколько реально есть).\n\n"
    "ВАЖНО:\n"
    "- Опирайся только на новости из списка, у каждой указана дата. Не выдумывай события.\n"
    "- Реальная текущая цена и её изменение даны отдельно — это факт. Не называй "
    "других уровней цены, процентов роста/падения и объёмов, чем в блоке «РЫНОК»: "
    "цифры в статьях могут относиться к другому моменту.\n"
    "- Если новости не про эту монету или не дают повода для вывода — sentiment 0, "
    "confidence 0.1-0.2, и прямо напиши, что значимых новостей нет."
)


@dataclass
class NewsResult:
    """Результат анализа новостей по символу."""
    sentiment: float            # -1.0 (медвежий) .. +1.0 (бычий)
    summary: str                # суммаризация на русском
    confidence: float           # 0.0 .. 1.0
    sources: list[str]          # URL источников
    key_events: list[str]       # ключевые события
    fetched_at: datetime        # когда получено
    via_search: bool            # True = через поиск (Tavily/Google), False = RSS


# CoinTelegraph tag-slug маппинг: тикер → RSS-тег.
_CT_TAG_MAP: dict[str, str] = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "BNB": "bnb", "XRP": "xrp", "ADA": "cardano",
    "DOGE": "dogecoin", "DOT": "polkadot", "AVAX": "avalanche",
    "MATIC": "polygon", "POL": "polygon", "LINK": "chainlink",
    "UNI": "uniswap", "ATOM": "cosmos", "NEAR": "near-protocol",
    "ARB": "arbitrum", "OP": "optimism", "APT": "aptos",
    "SUI": "sui", "TON": "toncoin", "TRX": "tron",
    "LTC": "litecoin", "INJ": "injective",
}


class NewsService:
    """Асинхронный сервис анализа новостей с кешем."""

    def __init__(self) -> None:
        # symbol -> (result, fetched_at)
        self._cache: dict[str, tuple[NewsResult | None, datetime]] = {}
        # Блокировка на символ, чтобы не слать параллельные запросы по одной монете.
        self._locks: dict[str, asyncio.Lock] = {}
        # aiohttp-сессия для RSS и API-вызовов.
        self._session: aiohttp.ClientSession | None = None
        # Клиент google-genai (создаётся лениво, только для gemini provider).
        self._genai_client = None
        self._genai_session = None
        # Circuit breaker для Tavily: если исчерпана квота (429/432), отключаем до указанного времени.
        self._tavily_disabled_until: float = 0.0

    # ─── Публичный API ────────────────────────────────────────

    async def get_sentiment(
        self,
        symbol: str,
        force: bool = False,
    ) -> NewsResult | None:
        """Получить сентимент новостей по символу."""
        if not self._is_available():
            return None

        coin = self._symbol_to_coin(symbol)

        # Кеш
        if not force:
            cached = self._cache.get(coin)
            if cached:
                result, fetched = cached
                age = (datetime.now(timezone.utc) - fetched).total_seconds()
                ttl = settings.news_cache_ttl if result is not None else 300
                if age < ttl:
                    return result

        # Блокировка: один запрос на символ за раз.
        lock = self._locks.setdefault(coin, asyncio.Lock())
        async with lock:
            if not force:
                cached = self._cache.get(coin)
                if cached:
                    result, fetched = cached
                    age = (datetime.now(timezone.utc) - fetched).total_seconds()
                    ttl = settings.news_cache_ttl if result is not None else 300
                    if age < ttl:
                        return result

            try:
                result = await self._fetch_and_analyze(coin)
            except Exception as exc:
                if force:
                    raise
                logger.warning("Ошибка анализа новостей для %s: %s", coin, exc)
                self._cache[coin] = (None, datetime.now(timezone.utc))
                return None

            if result is not None:
                self._cache[coin] = (result, datetime.now(timezone.utc))
            return result

    def get_cached(self, symbol: str) -> NewsResult | None:
        """Вернуть кешированный результат без сетевого запроса (для UI)."""
        if not self._is_available():
            return None
        coin = self._symbol_to_coin(symbol)
        cached = self._cache.get(coin)
        return cached[0] if cached else None

    def clear_cache(self, symbol: str | None = None) -> None:
        """Сбросить кеш (для кнопки «↻ Обновить»)."""
        if symbol is None:
            self._cache.clear()
        else:
            coin = self._symbol_to_coin(symbol)
            self._cache.pop(coin, None)

    async def close(self) -> None:
        """Закрыть HTTP-сессии при остановке приложения."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        if self._genai_session and not self._genai_session.closed:
            await self._genai_session.close()
        self._genai_session = None
        self._genai_client = None

    # ─── Внутренние ───────────────────────────────────────────

    def _is_available(self) -> bool:
        if not settings.news_enabled:
            return False
        provider = settings.news_provider
        if provider == "groq":
            return bool(settings.groq_api_key)
        if provider == "openrouter":
            return bool(settings.openrouter_api_key)
        if provider == "gemini":
            return bool(settings.gemini_api_key)
        return False

    @staticmethod
    def _symbol_to_coin(symbol: str) -> str:
        """'BTC/USDT' -> 'BTC'."""
        return symbol.split("/")[0].strip().upper()

    @property
    def _provider(self) -> str:
        return settings.news_provider

    async def _get_session(self) -> aiohttp.ClientSession:
        """aiohttp-сессия с надёжным резолвером (Windows DNS fix)."""
        if self._session is None or self._session.closed:
            from core.http_client import make_aiohttp_session
            self._session = make_aiohttp_session()
        return self._session

    # ─── Основной pipeline ────────────────────────────────────

    async def _fetch_and_analyze(self, coin: str) -> NewsResult | None:
        """Поиск новостей + LLM анализ.

        Стратегия:
          1. Tavily Search (если есть ключ) — приоритет: лучшая релевантность.
          2. RSS fallback (если Tavily недоступен или ключа нет).
          3. LLM анализ (OpenRouter или Gemini).
        """
        days = max(2, min(4, settings.news_lookback_days))
        news_text = ""
        sources: list[str] = []
        via_search = False

        # Шаг 1: Tavily — приоритетный источник (если ключ задан и квота не исчерпана).
        now_ts = asyncio.get_event_loop().time()
        if settings.tavily_api_key and now_ts >= self._tavily_disabled_until:
            try:
                news_text, sources = await self._search_tavily(coin, days)
                via_search = True
                if news_text:
                    logger.info("Tavily дал %d симв. новостей для %s", len(news_text), coin)
            except _QuotaExceeded as exc:
                self._tavily_disabled_until = now_ts + 3600
                logger.warning("Tavily лимит исчерпан: %s — отключаем на 1 час, fallback на RSS", exc)
            except Exception as exc:
                logger.warning("Tavily недоступен для %s: %s — fallback на RSS", coin, exc)

        # Шаг 2: RSS — fallback если Tavily недоступен или дал мало данных.
        if not news_text or len(news_text) < 300:
            logger.info("RSS fallback для %s (Tavily дал мало данных)", coin)
            rss_text, rss_src = await self._fetch_rss_news(coin)
            if rss_text:
                if news_text:
                    news_text += "\n\n--- Доп. RSS-источники ---\n" + rss_text
                else:
                    news_text = rss_text
                    via_search = False
                sources.extend(s for s in rss_src if s not in sources)

        if not news_text:
            # Свежих новостей именно по этой монете нет — это нормальная
            # ситуация, а не ошибка: нейтральный результат без вызова LLM.
            logger.info("Свежих новостей по %s за %d дн. нет — настроение нейтральное",
                        coin, days)
            return NewsResult(
                sentiment=0.0,
                summary=f"За последние {days} дня значимых новостей по {coin} не найдено.",
                confidence=0.0, sources=[], key_events=[],
                fetched_at=datetime.now(timezone.utc), via_search=via_search,
            )

        # Шаг 3: LLM анализ (с реальной ценой — чтобы модель не брала цифры
        # из статей за реальность).
        market = await _market_context(coin)
        prompt = self._build_analysis_prompt(coin, days, news_text, market)

        if self._provider == "groq":
            result = await self._call_groq(prompt, sources, via_search)
        elif self._provider == "openrouter":
            result = await self._call_openrouter(prompt, sources, via_search)
        elif self._provider == "gemini":
            result = await self._call_gemini(prompt, use_search=False)
        else:
            raise NewsError(f"Неизвестный провайдер: {self._provider}")
        return _drop_wrong_prices(result, market.get("price") if market else None, coin)

    # ─── Tavily Search ────────────────────────────────────────

    async def _search_tavily(
        self, coin: str, days: int
    ) -> tuple[str, list[str]]:
        """Поиск новостей через Tavily API.

        Returns (news_text, sources_urls).
        """
        session = await self._get_session()
        coin_name = _CT_TAG_MAP.get(coin, coin.lower())
        query = f"{coin_name.replace('-', ' ')} {coin} crypto news"

        # topic="news" обязателен: в общем режиме (по умолчанию) Tavily
        # игнорирует "days" и отдаёт статьи любого возраста — так в «свежие
        # новости за 3 дня» попадали статьи 2025 года с ценами того времени.
        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "topic": "news",
            "search_depth": "basic",
            "max_results": 10,
            "include_answer": False,
            "include_raw_content": False,
            "days": days,
            # Фильтруем на крипто-источники.
            "include_domains": [
                "cointelegraph.com",
                "coindesk.com",
                "theblock.co",
                "decrypt.co",
                "cryptoslate.com",
                "bitcoinist.com",
                "newsbtc.com",
                "ambcrypto.com",
            ],
        }

        timeout = aiohttp.ClientTimeout(total=20, connect=10)
        kwargs: dict = {"json": payload, "timeout": timeout}
        if settings.news_proxy:
            kwargs["proxy"] = settings.news_proxy
        async with session.post(
            "https://api.tavily.com/search",
            **kwargs,
        ) as resp:
            if resp.status in (429, 432):
                raise _QuotaExceeded("Tavily: исчерпан месячный лимит запросов.")
            if resp.status != 200:
                body = await resp.text(errors="ignore")
                raise NewsError(f"Tavily ошибка {resp.status}: {body[:200]}")
            data = await resp.json()

        # Парсинг результатов.
        results = data.get("results", [])
        if not results:
            return "", []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        items: list[tuple[str, str, str, datetime | None]] = []
        dropped_old = dropped_other = 0
        for r in results:
            title = (r.get("title") or "").strip()
            content = (r.get("content") or "").strip()
            url = (r.get("url") or "").strip()
            if not title:
                continue
            published = _parse_any_date(r.get("published_date")) or _date_from_url(url)
            if published is None or published < cutoff:
                dropped_old += 1
                continue
            if not _mentions_coin(f"{title} {content}", coin):
                dropped_other += 1
                continue
            items.append((title, content, url, published))
        if dropped_old or dropped_other:
            logger.info("Tavily %s: отброшено старых/без даты %d, не про монету %d, осталось %d",
                        coin, dropped_old, dropped_other, len(items))
        return _format_items(items)

    # ─── Groq LLM (1000+ req/день, Qwen 27B / GPT-OSS) ────────

    def _fallback_groq_model(self, current_model: str) -> str:
        """Запасная модель Groq если основная недоступна (404/429)."""
        fallbacks = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "groq/compound",
        ]
        if current_model in fallbacks:
            idx = fallbacks.index(current_model)
            return fallbacks[idx + 1] if idx + 1 < len(fallbacks) else ""
        return fallbacks[0]

    async def _call_groq(
        self,
        prompt: str,
        sources: list[str],
        via_search: bool,
    ) -> NewsResult | None:
        """Вызов LLM через Groq API (OpenAI-совместимый).

        Groq использует тот же формат что OpenAI: /openai/v1/chat/completions.
        Бесплатно: 1000+ req/день на qwen/qwen3.8-27b или openai/gpt-oss-120b.
        """
        session = await self._get_session()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": settings.groq_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 600,
            "top_p": 0.95,
        }
        _apply_groq_reasoning(payload)

        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=60, connect=15)
        kwargs: dict = {"json": payload, "headers": headers, "timeout": timeout}
        if settings.news_proxy:
            kwargs["proxy"] = settings.news_proxy

        try:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                **kwargs,
            ) as resp:
                if resp.status == 401:
                    raise NewsError(
                        "Неверный GROQ_API_KEY. Получи ключ: console.groq.com/keys"
                    )
                if resp.status in (404, 429):
                    tried = {settings.groq_model}
                    fb = self._fallback_groq_model(settings.groq_model)
                    while fb and fb not in tried:
                        tried.add(fb)
                        logger.info("Статус %d на Groq (%s), пробую fallback: %s", resp.status, payload["model"], fb)
                        payload["model"] = fb
                        _apply_groq_reasoning(payload)
                        async with session.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            **kwargs,
                        ) as resp2:
                            if resp2.status == 200:
                                data = await resp2.json()
                                choices = data.get("choices", [])
                                if choices:
                                    text = choices[0].get("message", {}).get("content", "")
                                    if text:
                                        return self._parse_llm_response(text, sources, via_search)
                            elif resp2.status not in (404, 429):
                                body = await resp2.text(errors="ignore")
                                raise NewsError(f"Groq ошибка {resp2.status}: {body[:200]}")
                        fb = self._fallback_groq_model(fb)

                    if resp.status == 404:
                        raise NewsError(
                            f"Модель '{settings.groq_model}' не найдена на Groq. "
                            "Укажи GROQ_MODEL=qwen/qwen3.8-27b в .env."
                        )
                    if resp.status == 429:
                        raise NewsError(
                            "Groq: исчерпан лимит запросов. "
                            "Подожди до завтра или смени провайдера в .env."
                        )

                if resp.status != 200:
                    body = await resp.text(errors="ignore")
                    raise NewsError(f"Groq ошибка {resp.status}: {body[:200]}")
                data = await resp.json()
        except aiohttp.ClientError as exc:
            raise NewsError(
                f"Groq недоступен (сеть/прокси): {exc}. "
                "Включи VPN или проверь NEWS_PROXY в .env."
            )

        choices = data.get("choices", [])
        if not choices:
            logger.warning("Groq вернул пустой choices")
            return None
        text = choices[0].get("message", {}).get("content", "")
        if not text:
            logger.warning("Groq вернул пустой текст")
            return None

        return self._parse_llm_response(text, sources, via_search)

    # ─── OpenRouter LLM ───────────────────────────────────────

    async def _call_openrouter(
        self,
        prompt: str,
        sources: list[str],
        via_search: bool,
    ) -> NewsResult | None:
        """Вызов LLM через OpenRouter API (OpenAI-совместимый).

        Если задан news_proxy — запросы идут через HTTP-прокси (обход geo-block).
        """
        session = await self._get_session()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": settings.openrouter_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 4096,
            "top_p": 0.95,
        }

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/kronos-trading",
            "X-Title": "Kronos Auto-Trading",
        }

        timeout = aiohttp.ClientTimeout(total=90, connect=15)
        kwargs: dict = {"json": payload, "headers": headers, "timeout": timeout}
        # Прокси для обхода geo-block OpenRouter (если задан в .env).
        if settings.news_proxy:
            kwargs["proxy"] = settings.news_proxy

        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            **kwargs,
        ) as resp:
            if resp.status == 429:
                # Пробуем запасные модели по цепочке, пока одна не сработает.
                tried = {settings.openrouter_model}
                fb = self._fallback_model(settings.openrouter_model)
                while fb and fb not in tried:
                    tried.add(fb)
                    logger.info("429 на %s, пробую fallback: %s", settings.openrouter_model, fb)
                    payload["model"] = fb
                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        **kwargs,
                    ) as resp2:
                        if resp2.status == 200:
                            result = await self._handle_openrouter_response(resp2, sources, via_search)
                            if result is not None:
                                return result
                        elif resp2.status != 429:
                            # Другая ошибка — пробрасываем.
                            return await self._handle_openrouter_response(resp2, sources, via_search)
                    fb = self._fallback_model(fb)
                raise NewsError(
                    "OpenRouter: исчерпан дневной лимит всех бесплатных моделей (50/день). "
                    "Подожди сброса (полночь UTC) или пополни $10 → 1000/день."
                )
            return await self._handle_openrouter_response(resp, sources, via_search)

    def _fallback_model(self, current_model: str) -> str:
        """Запасная модель если основная упала по 429.

        Цикл по убыванию размера: nemotron-super → nano-30b → nano-9b → gpt-oss
        """
        fallbacks = [
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "nvidia/nemotron-nano-9b-v2:free",
            "openai/gpt-oss-20b:free",
        ]
        if current_model in fallbacks:
            idx = fallbacks.index(current_model)
            return fallbacks[idx + 1] if idx + 1 < len(fallbacks) else ""
        return fallbacks[0]

    async def _handle_openrouter_response(self, resp, sources=None, via_search=False) -> NewsResult | None:
        """Обработка ответа OpenRouter (status + body → NewsResult)."""
        if resp.status == 401:
            raise NewsError(
                "Неверный OPENROUTER_API_KEY. Получи ключ: openrouter.ai/keys"
            )
        if resp.status == 404:
            raise NewsError(
                f"Модель '{settings.openrouter_model}' не найдена на OpenRouter. "
                "Проверь OPENROUTER_MODEL в .env."
            )
        if resp.status != 200:
            body = await resp.text(errors="ignore")
            raise NewsError(f"OpenRouter ошибка {resp.status}: {body[:200]}")
        data = await resp.json()

        # Извлекаем текст ответа.
        choices = data.get("choices", [])
        if not choices:
            logger.warning("OpenRouter вернул пустой choices")
            return None
        text = choices[0].get("message", {}).get("content", "")
        if not text:
            logger.warning("OpenRouter вернул пустой текст")
            return None

        return self._parse_llm_response(text, sources or [], via_search)

    # ─── Gemini (legacy) ──────────────────────────────────────

    async def _fetch_and_analyze_gemini(self, coin: str, days: int) -> NewsResult | None:
        """Legacy path: Gemini + Google Search → RSS fallback."""
        try:
            result = await self._call_gemini(
                self._build_search_prompt(coin, days), use_search=True
            )
            if result is not None:
                return result
            logger.info("Gemini search дал пустой ответ для %s, fallback на RSS", coin)
        except _QuotaExceeded:
            logger.info("Search-лимит исчерпан для %s, переключаюсь на RSS+Gemini", coin)
        except NewsError:
            raise

        # Fallback: RSS → Gemini без search.
        news_text, sources = await self._fetch_rss_news(coin)
        if not news_text:
            raise NewsError(
                "Search-лимит Gemini исчерпан, а RSS-источники недоступны. "
                "Попробуй позже или смени провайдер на OpenRouter."
            )
        result = await self._call_gemini(
            self._build_analysis_prompt(coin, days, news_text), use_search=False
        )
        if result is not None:
            if not result.sources:
                result.sources = sources
            return result
        return None

    def _get_genai_client(self):
        """Ленивая инициализация клиента google-genai (только для gemini provider)."""
        if self._genai_client is None:
            from google import genai
            from google.genai import types as gtypes
            from core.http_client import make_aiohttp_session

            self._genai_session = make_aiohttp_session()
            self._genai_client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=gtypes.HttpOptions(
                    aiohttp_client=self._genai_session,
                    timeout=30_000,
                ),
            )
        return self._genai_client

    async def _call_gemini(
        self, prompt: str, use_search: bool = True
    ) -> NewsResult | None:
        """Вызов Gemini через официальный SDK google-genai (async)."""
        from google import genai
        from google.genai import types
        from google.genai.errors import ClientError, ServerError

        client = self._get_genai_client()

        config_kwargs: dict = {
            "temperature": 0.3,
            "max_output_tokens": 4096,
            "top_p": 0.95,
            "system_instruction": _SYSTEM_PROMPT,
            "thinking_config": types.ThinkingConfig(thinking_budget=1024),
        }
        if use_search:
            config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = await client.aio.models.generate_content(
                model=settings.news_model,
                contents=prompt,
                config=config,
            )
        except ServerError as exc:
            raise NewsError(f"Gemini недоступен (ошибка сервера): {self._short(str(exc))}")
        except ClientError as exc:
            msg = str(exc)
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                if use_search:
                    raise _QuotaExceeded(msg)
                raise NewsError("Исчерпан лимит запросов Gemini.")
            if "404" in msg or "not found" in msg.lower():
                raise NewsError(f"Модель '{settings.news_model}' недоступна.")
            if "400" in msg and ("api key" in msg.lower() or "invalid" in msg.lower()):
                raise NewsError(
                    "Неверный GEMINI_API_KEY. Ключ начинается с 'AIza…'. "
                    "Получить: aistudio.google.com/apikey"
                )
            raise NewsError(f"Gemini отклонил запрос: {self._short(msg)}")
        except asyncio.TimeoutError:
            raise NewsError("Gemini не ответил за 30 сек (таймаут).")
        except Exception as exc:
            logger.warning("Gemini непредвиденная ошибка: %s", exc, exc_info=True)
            raise NewsError(f"Непредвиденная ошибка Gemini: {self._short(str(exc))}")

        # Парсинг SDK-ответа.
        try:
            text = response.text or ""
        except Exception:
            text = ""
        if not text and getattr(response, "candidates", None):
            for cand in response.candidates:
                content = getattr(cand, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        if getattr(part, "text", None):
                            text += part.text
        if not text:
            return None

        # Источники из grounding_metadata (только для search-режима).
        gemini_sources: list[str] = []
        if use_search and getattr(response, "candidates", None):
            gemini_sources = self._extract_sources_sdk(response)

        return self._parse_llm_response(text, gemini_sources, via_search=use_search)

    @staticmethod
    def _extract_sources_sdk(response) -> list[str]:
        """Достать URL из grounding_metadata SDK-ответа."""
        sources: list[str] = []
        for cand in getattr(response, "candidates", []) or []:
            gm = getattr(cand, "grounding_metadata", None)
            if not gm:
                continue
            chunks = getattr(gm, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web and getattr(web, "uri", None):
                    sources.append(web.uri)
        seen: set[str] = set()
        unique: list[str] = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique[:8]

    # ─── Промпты ──────────────────────────────────────────────

    @staticmethod
    def _build_search_prompt(coin: str, days: int) -> str:
        """Промпт для Gemini Search mode."""
        return (
            f"Проанализируй последние новости по криптовалюте {coin} за "
            f"последние {days} дня. Найди значимые события: партнёрства, "
            f"листинги на биржах, регуляторные новости, обновления сети, "
            f"крупные движения капитала, взломы или иски. Оцени настроение "
            f"рынка по этой монете и дай ответ в формате JSON."
        )

    @staticmethod
    def _build_analysis_prompt(
        coin: str, days: int, news_text: str, market: dict | None = None,
    ) -> str:
        """Промпт для анализа уже собранных новостей (Tavily или RSS)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if market:
            parts = [f"текущая цена {coin}: ${_fmt_price(market['price'])}"]
            for label, key in (("24 ч", "ch24"), ("3 дня", "ch72"), ("7 дней", "ch168")):
                if market.get(key) is not None:
                    parts.append(f"за {label}: {market[key]:+.1f}%")
            market_line = "РЫНОК (факт, Binance): " + ", ".join(parts) + "\n\n"
        else:
            market_line = ""
        return (
            f"Сегодня {today} (UTC). Ниже — новости по криптовалюте {coin} "
            f"за последние {days} дня, у каждой указана дата публикации. "
            f"Оцени их влияние на настроение рынка по {coin}: события сети, "
            f"листинги, регуляторные риски, движения капитала. "
            f"Дай ответ в формате JSON.\n\n"
            f"{market_line}"
            f"НОВОСТИ:\n{news_text}"
        )

    # ─── Парсинг LLM-ответа ──────────────────────────────────

    def _parse_llm_response(
        self,
        text: str,
        sources: list[str],
        via_search: bool,
    ) -> NewsResult | None:
        """Общий парсер JSON-ответа для OpenRouter и Gemini."""
        sentiment_data = _extract_json(text)
        if not sentiment_data:
            logger.warning("Не удалось распарсить sentiment JSON: %s", text[:300])
            return None

        try:
            sentiment = float(sentiment_data.get("sentiment", 0.0))
        except (TypeError, ValueError):
            sentiment = 0.0
        sentiment = max(-1.0, min(1.0, sentiment))

        try:
            confidence = float(sentiment_data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        summary = str(sentiment_data.get("summary", "")).strip()
        key_events = sentiment_data.get("key_events", [])
        if not isinstance(key_events, list):
            key_events = []
        key_events = [str(e).strip() for e in key_events if e][:5]

        return NewsResult(
            sentiment=sentiment,
            summary=summary or "Новости проанализированы.",
            confidence=confidence,
            sources=sources,
            key_events=key_events,
            fetched_at=datetime.now(timezone.utc),
            via_search=via_search,
        )

    # ─── RSS fallback ─────────────────────────────────────────

    async def _fetch_rss_news(self, coin: str) -> tuple[str, list[str]]:
        """Загрузить свежие новости из бесплатных RSS-источников."""
        days = max(2, min(4, settings.news_lookback_days))
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        session = await self._get_session()
        coin_lower = coin.lower()

        ct_tag = _CT_TAG_MAP.get(coin, coin_lower)
        feeds = [
            f"https://cointelegraph.com/rss/tag/{ct_tag}",
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://decrypt.co/feed",
            "https://bitcoinist.com/feed/",
        ]
        items: list[tuple[str, str, str, datetime | None]] = []
        tag_feed = feeds[0]
        for feed_url in feeds:
            try:
                timeout = aiohttp.ClientTimeout(total=10, connect=8)
                kwargs: dict = {"timeout": timeout}
                if settings.news_proxy:
                    kwargs["proxy"] = settings.news_proxy
                async with session.get(feed_url, **kwargs) as resp:
                    if resp.status != 200:
                        continue
                    xml = await resp.text(errors="ignore")
                for item_m in re.finditer(r"<item>(.*?)</item>", xml, re.DOTALL):
                    block = item_m.group(1)
                    title = re.search(
                        r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", block, re.DOTALL
                    )
                    desc = re.search(
                        r"<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>",
                        block, re.DOTALL,
                    )
                    link = re.search(r"<link>(.*?)</link>", block, re.DOTALL)
                    pub = re.search(r"<pubDate>(.*?)</pubDate>", block, re.DOTALL)
                    t = re.sub(r"<!\[CDATA\[|\]\]>", "", title.group(1)).strip() if title else ""
                    t = t.replace("\n", " ")
                    d_raw = re.sub(r"<!\[CDATA\[|\]\]>", "", desc.group(1)).strip() if desc else ""
                    d_clean = re.sub(r"<[^>]+>", " ", d_raw).strip()
                    l_raw = (link.group(1).strip() if link else "")
                    if not l_raw:
                        link_sc = re.search(r'<link\s+[^>]*/>', block)
                        if link_sc:
                            href = re.search(r'href="([^"]+)"', link_sc.group(0))
                            if href:
                                l_raw = href.group(1)
                    l = re.sub(r"<!\[CDATA\[|\]\]>", "", l_raw).strip()
                    pub_dt = _parse_rss_date(pub.group(1).strip()) if pub else None
                    if not t:
                        continue
                    if pub_dt is None or pub_dt < cutoff:
                        continue
                    # Общие ленты (не тег монеты) — только статьи, где монета
                    # упомянута. Раньше этот фильтр не срабатывал никогда, и в
                    # «новости по DOT» попадали общие новости рынка.
                    if feed_url != tag_feed and not _mentions_coin(f"{t} {d_clean}", coin):
                        continue
                    items.append((t, d_clean, l, pub_dt))
            except Exception:
                logger.debug("RSS-источник недоступен: %s", feed_url, exc_info=True)
                continue
            if len(items) >= 15:
                break

        # Дедуп по заголовку.
        seen: set[str] = set()
        unique: list[tuple[str, str, str, datetime | None]] = []
        for t, d, l, dt in items:
            key = t.lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            unique.append((t, d, l, dt))
        return _format_items(unique[:12])

    # ─── Утилиты ─────────────────────────────────────────────

    @staticmethod
    def _short(text: str, limit: int = 200) -> str:
        return text if len(text) <= limit else text[:limit] + "…"


class _QuotaExceeded(Exception):
    """Внутренний сигнал: исчерпан search-лимит Gemini → fallback на RSS."""


# ─── JSON extraction (общая утилита) ────────────────────────

def _apply_groq_reasoning(payload: dict) -> None:
    """Отключить/урезать «размышления» у reasoning-моделей Groq.

    qwen3.8 по умолчанию думает (reasoning_effort="default"): размышления
    съедают лимит токенов (free: ~8K TPM / 200K TPD) и при max_tokens=600
    обрезают JSON. Для qwen — режим instruct ("none"), для gpt-oss — "low"
    (у gpt-oss значения "none" нет).
    """
    model = str(payload.get("model", ""))
    payload.pop("reasoning_effort", None)
    if model.startswith("qwen/"):
        payload["reasoning_effort"] = "none"
    elif model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"


def _extract_json(text: str) -> dict | None:
    """Найти первый JSON-объект в тексте и распарсить."""
    # Убираем markdown-обёртку ```json ... ```.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Ищем подстроку от первой { до конца текста.
    start = text.find("{")
    if start == -1:
        return None
    candidate = text[start:].strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Восстановление обрезанного JSON.
    return _repair_json(candidate)


def _repair_json(text: str) -> dict | None:
    """Попытаться восстановить обрезанный JSON."""
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    repaired = text
    if in_str:
        repaired += '"'
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


# Дополнительные названия монет для проверки «статья про эту монету».
_EXTRA_NAMES: dict[str, list[str]] = {
    "ATOM": ["cosmos hub"], "POL": ["matic"], "BNB": ["binance coin", "bnb chain"],
    "NEAR": ["near protocol"], "TON": ["the open network"],
}


def _mentions_coin(text: str, coin: str) -> bool:
    """Статья действительно про монету: тикер (как слово) или её название.

    Тикер ищется с учётом регистра и границ слова: иначе «DOT» находился бы
    в «dot-com», «ATOM» — в «atomic», «OP» — где угодно.
    """
    name = _CT_TAG_MAP.get(coin, coin.lower()).replace("-", " ")
    low = text.lower()
    names = [name, *_EXTRA_NAMES.get(coin, [])]
    if any(re.search(rf"\b{re.escape(n)}\b", low) for n in names if len(n) > 3):
        return True
    return re.search(rf"(?<![A-Za-z]){re.escape(coin)}(?![A-Za-z])", text) is not None


def _parse_any_date(value) -> datetime | None:
    """Дата из ответа Tavily (RFC 2822 или ISO) → aware UTC."""
    if not value:
        return None
    text = str(value).strip()
    dt = _parse_rss_date(text)
    if dt is None:
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _date_from_url(url: str) -> datetime | None:
    """Дата из URL вида …/2025/12/23/… (coindesk и др.) — страховка, если
    у результата нет published_date."""
    m = re.search(r"/(20\d\d)/(\d\d)/(\d\d)/", url or "")
    if not m:
        return None
    try:
        return datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
    except ValueError:
        return None


def _format_items(items: list[tuple[str, str, str, datetime | None]]) -> tuple[str, list[str]]:
    """Список новостей для LLM: каждая с датой публикации (UTC)."""
    items = sorted(items, key=lambda it: it[3] or datetime.min.replace(tzinfo=timezone.utc),
                   reverse=True)
    lines: list[str] = []
    sources: list[str] = []
    for i, (title, desc, url, published) in enumerate(items, 1):
        date = published.strftime("%Y-%m-%d %H:%M") if published else "дата неизвестна"
        lines.append(f"{i}. [{date} UTC] {title}")
        if desc and len(desc) > 20:
            lines.append(f"   {desc[:350]}")
        if url and url not in sources:
            sources.append(url)
    return "\n".join(lines), sources


def _fmt_price(p: float) -> str:
    return f"{p:,.2f}" if p >= 100 else (f"{p:.4f}" if p >= 1 else f"{p:.5f}")


async def _market_context(coin: str) -> dict | None:
    """Реальная цена и её изменение за 24 ч / 3 дня / 7 дней из свечей 1h в БД."""
    try:
        from sqlalchemy import select

        from db.database import get_session
        from db.models import Candle, Coin

        async with get_session() as db:
            coin_row = (await db.execute(
                select(Coin).where(Coin.symbol == f"{coin}/USDT")
            )).scalar_one_or_none()
            if coin_row is None:
                return None
            rows = (await db.execute(
                select(Candle.close)
                .where(Candle.coin_id == coin_row.id, Candle.tf == "1h")
                .order_by(Candle.timestamp.desc())
                .limit(169)
            )).scalars().all()
        if not rows:
            return None
        price = float(rows[0])
        ctx: dict = {"price": price}
        for key, h in (("ch24", 24), ("ch72", 72), ("ch168", 168)):
            ctx[key] = (price / float(rows[h]) - 1) * 100 if len(rows) > h and rows[h] else None
        return ctx
    except Exception as exc:
        logger.debug("Нет рыночного контекста для %s: %s", coin, exc)
        return None


_PRICE_RE = re.compile(r"\$\s?(\d[\d,\s]*(?:[.,]\d+)?)")


def _has_wrong_price(text: str, price: float) -> bool:
    """В тексте есть цена в $, отличающаяся от реальной больше чем в 1.5 раза."""
    for m in _PRICE_RE.finditer(text):
        raw = m.group(1).replace(" ", "").replace("\u202f", "")
        if raw.count(",") and raw.count("."):
            raw = raw.replace(",", "")
        else:
            raw = raw.replace(",", ".")
        try:
            v = float(raw)
        except ValueError:
            continue
        # Крупные суммы ($2.31 млн, $500M) — это объёмы, а не цена монеты.
        tail = text[m.end():m.end() + 6].lower()
        if any(w in tail for w in ("млн", "млрд", "тыс", "m ", "b ", "k ", "mln", "bn")):
            continue
        if v > 0 and not (price / 1.5 <= v <= price * 1.5):
            return True
    return False


def _drop_wrong_prices(result: "NewsResult | None", price: float | None, coin: str):
    """Убрать из сводки фразы с ценами, не совпадающими с реальной.

    Страховка на случай, если модель всё-таки взяла цифры из статей.
    """
    if result is None or not price:
        return result
    sentences = re.split(r"(?<=[.!?])\s+", result.summary)
    kept = [s for s in sentences if not _has_wrong_price(s, price)]
    events = [e for e in result.key_events if not _has_wrong_price(e, price)]
    removed = (len(sentences) - len(kept)) + (len(result.key_events) - len(events))
    if removed:
        logger.warning("Новости %s: убрано %d фраз с ценой, не совпадающей с реальной ($%s)",
                       coin, removed, _fmt_price(price))
        result.summary = " ".join(kept) or "Новости проанализированы."
        result.key_events = events
    return result


def _parse_rss_date(date_str: str) -> datetime | None:
    """Распарсить дату из RSS pubDate (RFC822 и вариации)."""
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ─── Синглтон ────────────────────────────────────────────────

_news_svc: NewsService | None = None


def get_news_service() -> NewsService:
    """Синглтон NewsService (создаётся при первом обращении)."""
    global _news_svc
    if _news_svc is None:
        _news_svc = NewsService()
    return _news_svc
