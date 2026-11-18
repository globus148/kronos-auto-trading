"""Тесты: новости только свежие, только про монету, без чужих цен.

Раньше Tavily вызывался без topic="news" и игнорировал "days": в «новости
за 3 дня» по DOT/ATOM попадали статьи 2025 года, а модель пересказывала
их цены ($1.76, $4.55) как текущие.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import core.news_service as ns
from core.news_service import (
    NewsResult,
    NewsService,
    _date_from_url,
    _drop_wrong_prices,
    _has_wrong_price,
    _mentions_coin,
)


def _rfc(dt: datetime) -> str:
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


class _Resp:
    def __init__(self, status=200, json_data=None, text=""):
        self.status, self._json, self._text = status, json_data, text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._json

    async def text(self, errors=None):
        return self._text


class _Session:
    def __init__(self, post_json=None, feeds=None):
        self.post_json, self.feeds, self.posted = post_json, feeds or {}, []

    def post(self, url, **kw):
        self.posted.append(kw.get("json"))
        return _Resp(json_data=self.post_json)

    def get(self, url, **kw):
        return _Resp(status=200 if url in self.feeds else 404, text=self.feeds.get(url, ""))


class TestHelpers:
    def test_mentions_coin(self):
        assert _mentions_coin("Polkadot launches JAM upgrade", "DOT")
        assert _mentions_coin("DOT slips 3% as markets fall", "DOT")
        assert not _mentions_coin("The dot-com bubble and AI stocks", "DOT")
        assert _mentions_coin("Cosmos Hub votes on ATOM inflation", "ATOM")
        assert not _mentions_coin("Atomic swaps come to Bitcoin", "ATOM")
        assert not _mentions_coin("Bitcoin ETF inflows hit record", "ATOM")

    def test_date_from_url(self):
        d = _date_from_url("https://www.coindesk.com/markets/2025/12/23/polkadot-s-dot-slips")
        assert d == datetime(2025, 12, 23, tzinfo=timezone.utc)
        assert _date_from_url("https://ambcrypto.com/cosmos-crypto-defies") is None

    def test_wrong_price_detection(self):
        assert _has_wrong_price("DOT теряет поддержку $1,76", 0.73)
        assert not _has_wrong_price("DOT держится около $0.72", 0.73)
        assert not _has_wrong_price("объём $2.31 млн", 0.73)

    def test_drop_wrong_prices(self):
        r = NewsResult(
            sentiment=-0.4,
            summary="Сеть обновилась. Цена пробила $4.55 и выросла. Регуляторы молчат.",
            confidence=0.6, sources=[], key_events=["Пробой $4.34", "Обновление сети"],
            fetched_at=datetime.now(timezone.utc), via_search=True,
        )
        out = _drop_wrong_prices(r, 1.33, "ATOM")
        assert "$4.55" not in out.summary and "Сеть обновилась." in out.summary
        assert out.key_events == ["Обновление сети"]


class TestTavilyFreshness:
    @pytest.mark.asyncio
    async def test_old_and_offtopic_articles_are_dropped(self, monkeypatch):
        now = datetime.now(timezone.utc)
        results = [
            {"title": "Polkadot's DOT slips 4.5% as token underperforms", "content": "DOT fell to $1.76",
             "url": "https://www.coindesk.com/markets/2025/12/23/polkadot-s-dot-slips",
             "published_date": _rfc(datetime(2025, 12, 23, tzinfo=timezone.utc))},
            {"title": "Polkadot DAO approves treasury proposal", "content": "The Polkadot community voted",
             "url": "https://example.com/a", "published_date": _rfc(now - timedelta(hours=10))},
            {"title": "Bitcoin ETF flows turn positive", "content": "BTC funds saw inflows",
             "url": "https://example.com/b", "published_date": _rfc(now - timedelta(hours=5))},
            {"title": "Polkadot no date", "content": "Polkadot", "url": "https://example.com/c"},
        ]
        sess = _Session(post_json={"results": results})
        svc = NewsService()

        async def get_session():
            return sess
        monkeypatch.setattr(svc, "_get_session", get_session)
        monkeypatch.setattr(ns.settings, "tavily_api_key", "k", raising=False)

        text, sources = await svc._search_tavily("DOT", 3)
        assert sess.posted[0]["topic"] == "news" and sess.posted[0]["days"] == 3
        assert "treasury" in text and "UTC]" in text
        assert "1.76" not in text and "Bitcoin ETF" not in text and "no date" not in text
        assert sources == ["https://example.com/a"]


class TestNoNewsIsNeutral:
    @pytest.mark.asyncio
    async def test_no_fresh_news_returns_neutral_without_llm(self, monkeypatch):
        svc = NewsService()
        monkeypatch.setattr(ns.settings, "tavily_api_key", "", raising=False)

        async def no_rss(coin):
            return "", []
        monkeypatch.setattr(svc, "_fetch_rss_news", no_rss)

        async def boom(*a, **k):
            raise AssertionError("LLM не должен вызываться")
        monkeypatch.setattr(svc, "_call_groq", boom)
        monkeypatch.setattr(svc, "_call_openrouter", boom)
        monkeypatch.setattr(svc, "_call_gemini", boom)

        r = await svc._fetch_and_analyze("ATOM")
        assert r.sentiment == 0.0 and r.confidence == 0.0
        assert "не найдено" in r.summary


class TestRssFilter:
    @pytest.mark.asyncio
    async def test_general_feeds_keep_only_coin_articles(self, monkeypatch):
        now = datetime.now(timezone.utc)

        def item(title, dt):
            return (f"<item><title>{title}</title><link>https://x/{abs(hash(title))}</link>"
                    f"<pubDate>{_rfc(dt)}</pubDate><description>text text text text text</description></item>")
        general = "<rss>" + item("Bitcoin hits new high", now) + item("Cosmos ATOM staking update", now) \
            + item("Cosmos old news", now - timedelta(days=30)) + "</rss>"
        sess = _Session(feeds={"https://cointelegraph.com/rss": general})
        svc = NewsService()

        async def get_session():
            return sess
        monkeypatch.setattr(svc, "_get_session", get_session)

        text, sources = await svc._fetch_rss_news("ATOM")
        assert "Cosmos ATOM staking update" in text
        assert "Bitcoin hits new high" not in text
        assert "old news" not in text
        assert len(sources) == 1

    def test_prompt_contains_real_price(self):
        p = NewsService._build_analysis_prompt("DOT", 3, "1. [..] x", {"price": 0.729, "ch24": -1.2, "ch72": -2.3, "ch168": -12.0})
        assert "$0.72900" in p and "-12.0%" in p and "РЫНОК" in p
