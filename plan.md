# План: приложение автоторговли криптовалютой на базе Kronos (прототип)

## 1. Цель и скоуп

Создать **прототип** веб-приложения автоторговли криптовалютой: два режима (тестовый/реальный),
режимы торговли (авто/ручной), выбор монеты, аналитика. Ядро предсказания — **Kronos**
(foundation model для OHLCV) локально на RTX 3060, поверх гибридный сигнальный слой
(Kronos + классические индикаторы + математика рисков).

**Скоуп прототипа:** только **тестовый режим** (paper trading), spot long + кэш-позиции,
Binance через CCXT, таймфреймы 1h/4h. Реальный режим — архитектурная заглушка.

---

## 2. Стек

| Слой            | Технология                                                  |
| --------------- | ----------------------------------------------------------- |
| Backend         | Python 3.11 + FastAPI (async + WebSocket)                   |
| UI              | Jinja2 + HTMX + Plotly (server-rendered, как у Kronos)      |
| ML-ядро         | Kronos-base (PyTorch CUDA), RTX 3060                        |
| Биржа/данные    | CCXT async → Binance (публичное API, без ключей)            |
| БД              | SQLite + SQLAlchemy (прототип) → Postgres (прод)            |
| Цикл            | APScheduler                                                 |
| Запуск          | docker-compose                                              |

---

## 3. Архитектура

```
Браузер (Jinja2 + HTMX + Plotly)
        │ HTTP + WebSocket
FastAPI ──┬── routes/ (dashboard, coin, trade, analytics, backtest)
         ├── auth (флаг тестовый/реальный в сессии)
         └── ws (push PnL/позиций)
   │
   ├─ MarketService (CCXT: OHLCV, тикер)
   ├─ PredictionService (Kronos + кэш)
   ├─ Indicators (RSI/MACD/EMA/ATR/BB/ADX)
   ├─ StrategyEngine (сигналы entry/exit — математика)
   ├─ RiskManager (sizing, ATR стоп)
   ├─ PortfolioEngine (сделки, PnL, equity)
   ├─ DataLifecycleManager  ← (раздел 6)
   └─ TradingLoop (APScheduler)
   │
SQLite: coin, candle, position, trade, equity_point,
        prediction, indicator_snapshot, strategy_config
```

### Модули (пакеты)

- `app/` — FastAPI: `main.py`, `routes/`, `auth.py`, `ws.py`, `templates/`, `static/`
- `core/market_service.py` — обёртка над CCXT (OHLCV, тикер, стакан)
- `core/data_lifecycle.py` — cold/warm старт, pruning, incremental-обновление (раздел 6)
- `core/prediction_service.py` — загрузка Kronos, кэш прогнозов, нормализация
- `core/indicators.py` — RSI, MACD, EMA, ATR, Bollinger, ADX (pandas/numpy)
- `core/strategy_engine.py` — аналитическая модель → сигналы entry/exit
- `core/risk_manager.py` — размер позиции, стоп-лосс, лимиты
- `core/portfolio_engine.py` — журнал сделок, PnL, комиссии, equity
- `core/trading_loop.py` — APScheduler-цикл (тик на закрытии свечи каждого ТФ)
- `core/backtester.py` — исторический бэктест стратегии
- `db/` — SQLAlchemy-модели, миграции
- `kronos/` — vendored `model/` из Kronos-репо
- `config.py` — настройки (таймфреймы, комиссии, параметры Kronos, глубины данных)

---

## 4. Математическая модель аналитики (ядро)

### 4.1. Порог безубыточности с комиссией

```
commission_rate = 0.001   (0.1% Binance spot)
slippage_rate   = 0.0005  (0.05% проскальзывание)
round_trip_cost = 2 * (commission_rate + slippage_rate) = 0.003   (0.3%)
min_expected_move = round_trip_cost + margin_of_safety(0.3%) = 0.006   (0.6%)
```

Ни один сигнал Kronos не вызовет сделку, если прогнозируемое движение < `min_expected_move`.
Это гарантия «выхода в плюс даже с комиссией».

### 4.2. Входные данные (feature vector)

1. **Прогноз Kronos** (N свечей вперёд, вероятностный, `sample_count=3`):
   - `pred_return_N = predicted_close[t+N] / close[t] − 1` — ожидаемая доходность на горизонте
   - `pred_max_high = max(predicted_high)` — ожидаемый максимум
   - `pred_min_low  = min(predicted_low)` — ожидаемый минимум
   - `pred_path_slope` — наклон линейной регрессии по прогнозу (ускорение/замедление)
2. **Индикаторы** (окно 14–26 под ТФ):
   RSI(14), MACD(12,26,9), EMA(20), EMA(50), ATR(14), Bollinger(20,2), ADX(14)

### 4.3. Композитный сигнал входа (ансамбль ∈ [0,1])

```
S_entry = 0.25 · f_trend    (EMA20 > EMA50 + наклон)
        + 0.30 · f_kronos   (pred_return_N ≥ min_expected_move AND slope > 0)
        + 0.20 · f_momentum (MACD пересечение вверх, гистограмма растёт)
        + 0.15 · f_rsi      (RSI ∈ [40,70] — не перекуплен, но есть сила)
        + 0.10 · f_pullback (откат к EMA20 из перепроданности)

веса по умолчанию: Kronos доминирует.

Вход (long): S_entry ≥ 0.65
             AND pred_return_N ≥ min_expected_move      # комиссионный фильтр
             AND ADX ≥ 20                               # есть тренд, не флэт
             AND risk/reward ≥ 2.0
             AND expected_pnl > 0                       # см. 4.6
```

### 4.4. Условие выхода (по первому сработавшему)

1. **Take-profit**: цена ≥ `entry_price · (1 + target_pct)` (target из `pred_max_high`
   с дисконтом / R-multiple от риска).
2. **Trailing stop** (активируется после 1R прибыли): стоп следует за ценой на дистанции
   `k · ATR` (k ≈ 2). Защищает накопленную прибыль.
3. **ATR stop-loss**: цена ≤ `entry_price − k_sl · ATR` (k_sl ≈ 1.5).
4. **Signal reversal**: Kronos развернулся (`pred_path_slope < 0` AND
   `pred_return_N < −min_expected_move`).
5. **Time stop**: позиция держится дольше `max_hold_periods` (напр. 48 свечей) → закрыть по рынку.

### 4.5. Размер позиции — Fixed-Fractional + ATR

```
risk_per_trade = 0.01 · equity           # 1% капитала
stop_distance  = k_sl · ATR[t]           # в цене
stop_pct       = stop_distance / close[t]
position_value = risk_per_trade / stop_pct
qty            = position_value / close[t]   # max 30% equity, не больше доступного USDT
```

Так максимальный убыток по сделке = риск 1% капитала независимо от волатильности монеты.

### 4.6. Фильтр безубыточности перед входом

```
best_case_pnl  = qty · (target_price − entry_price) − round_trip_cost_value
worst_case_pnl = −risk_per_trade                      # по стопу
expected_pnl   = p_win · best_case_pnl − (1 − p_win) · risk_per_trade
```

`p_win` калибруется из истории бэктестов по `S_entry`. **Вход только если `expected_pnl > 0`.**

### 4.7. Метрики аналитики (для UI)

- Equity curve, drawdown curve
- Total/PnL%, win rate, profit factor, Sharpe, Sortino, max drawdown
- Средняя сделка, средняя просадка, expectancy
- Распределение сделок по монетам/ТФ
- Точность прогноза Kronos (MAE/направление) vs факт
- Вклад комиссии в общий PnL

---

## 5. Функционал (фичи прототипа)

- **Вход — выбор режима.** Тестовый → старт **$100 USDT** (виртуальный), все сделки paper.
  Реальный → заглушка «скоро» (требует API-ключи Binance, не реализуем).
- **Дашборд:** equity-кривая, баланс, PnL за период, список монет, открытые позиции.
- **Экран монеты:** график свечей (Plotly) + наложение прогноза Kronos (полупрозрачный конус),
  индикаторы (RSI/MACD/EMA под графиком), датчики `S_entry`/`S_exit`, выбор ТФ (1h/4h),
  кнопки **Авто ВКЛ/ВЫКЛ** и ручные **Купить/Продать**.
- **Аналитика торговли:** таблица сделок (вход/выход/причина выхода/комиссия/PnL),
  сводные метрики (4.7), сравнение PnL с/без учёта комиссии, тепловая карта по монетам.
- **Бэктест (встроенный):** выбор монеты, периода, ТФ, параметров стратегии → отчёт + equity.
- **Настройки:** `risk_fraction`, `entry_threshold`, `k_sl`, `k_tp`, таймфреймы, список монет,
  параметры Kronos (model size, T, top_p, sample_count).

---

## 6. Управление данными (Data Lifecycle)

**Принцип:** данные OHLCV — расходный материал. Хранится только нужное для (1) контекста Kronos,
(2) индикаторов, (3) бэктеста. Всё устаревшее/лишнее удаляется. БД не растёт бесконечно.

### 6.1. Конфигурация глубины

```
kronos_context   = 512   # макс. контекст Kronos-base/small
indicator_warmup = 200   # EMA50, MACD, ATR
live_buffer      = 100   # запас свежих свечей
backtest_window  = 2000  # настраиваемый горизонт бэктеста

required_history(coin, tf) = max(
    kronos_context + live_buffer,     # ≈ 612 (прогноз)
    indicator_warmup + live_buffer,   # ≈ 300 (индикаторы)
    backtest_window                  # 2000 (бэктест)
)                                    # → итого ~2000 свечей на старте
retention_size = required_history · 1.1   # скользящее окно хранения
```

Для 4h × 2000 ≈ 333 дня истории; для 1h × 2000 ≈ 83 дня.

### 6.2. Сценарий A — первичный запуск (cold start)

1. Приложение стартует → для каждой включённой монеты и ТФ:
2. Проверяем `candle` в БД: есть ли нужная глубина и свежесть.
3. Если нет → CCXT `fetch_ohlcv` пагинацией (по 500–1000 свечей) скачиваем
   `required_history` свечей назад от now.
4. Bulk-insert в `candle` (индекс `coin_id, tf, timestamp DESC`).
5. Лог: `Downloaded 2000 candles for BTC/USDT 4h`.

### 6.3. Сценарий B — перезапуск (warm start)

1. Для каждой монеты/ТФ: проверяем последнюю свечу в БД.
2. `gap = now − last_candle_ts`.
3. Если `gap > 2 свечи` (отставание) ИЛИ глубина < `required_history`:
   - **Малый gap** → incremental fetch только дельты, дополнить.
   - **Большой gap / дыры / фрагментация** → полный пересчёт: `DELETE` всех свечей
     этой пары/ТФ и cold-start скачивание заново. Политика: «лучше свежие и ровные,
     чем старые с дырами».

### 6.4. Сценарий C — удаление старых неэффективных свечей (pruning)

После каждой вставки новых свечей:

```
prune_candles(coin, tf):
  keep_from = now − retention_size · interval
  DELETE FROM candle
   WHERE coin_id = ? AND tf = ? AND timestamp < keep_from
```

Держим скользящее окно свежайших `retention_size` свечей. Старые «неэффективные»
(вне контекста Kronos и вне бэктеста) удаляются → БД не захламляется.
Раз в сутки `VACUUM` SQLite для освобождения места.

### 6.5. Сценарий D — live-добавление по необходимости

Торговый цикл на закрытии свечи:

1. Fetch последней закрытой свечи (1 шт).
2. Insert в `candle`.
3. `prune_candles()` — если превысили retention, удалить лишнюю снизу.
4. Таблица `candle` = ровно скользящее окно. Kronos и индикаторы читают последние N.
5. Если авто-торговля по монете выключена юзером → цикл ставится на паузу, лишнее не качается.

### 6.6. Оптимизации памяти/БД

- OHLCV: ~7 float × ~2000 свечей × ~10 монет × 2 ТФ ≈ 40k строк — мало для SQLite,
  но retention держит чистоту.
- Индекс `(coin_id, tf, timestamp DESC)` — быстрый «последние N».
- Кэш прогнозов Kronos — отдельная таблица `prediction` с TTL, тоже прунится.
- Реже торгуемые монеты → урезанное окно (только `kronos_context`).

### 6.7. Контроль качества данных

- Детект дыр в таймстампах → дозагрузка.
- Детект аномалий (нулевой объём/OHLC) → флаг, изоляция.
- Сверка `close` последней свечи с live-тикером → детект рассинхрона с биржей.

---

## 7. Схема данных (SQLite, SQLAlchemy)

- `coin` (id, symbol, enabled, default_tf)
- `candle` (id, coin_id, tf, timestamp, open, high, low, close, volume) — **retention-таблица**
- `session` (id, mode [paper/live], initial_balance, created_at)
- `position` (id, session_id, coin_id, side, qty, entry_price, entry_at, stop_price, target_price, status)
- `trade` (id, position_id, side, qty, price, fee, slippage, pnl, reason, executed_at)
- `equity_point` (id, session_id, timestamp, equity, cash, positions_value)
- `prediction` (id, coin_id, tf, predicted_at, horizon, payload_json, actual_return_later) — TTL-кэш Kronos
- `indicator_snapshot` (id, coin_id, tf, timestamp, rsi, macd, atr, ..., s_entry, s_exit)
- `strategy_config` (key, value_json)

---

## 8. Торговый цикл (live loop)

`APScheduler` запускает задачу на закрытии свечи выбранного ТФ:

1. **`DataLifecycleManager.update(coin, tf)`** — fetch + insert + prune (раздел 6.5).
2. Считаем индикаторы → `indicator_snapshot`.
3. Kronos прогноз (кэш по `coin+tf+timestamp`) → `prediction`.
4. Считаем `S_entry` / `S_exit`.
5. Для каждой открытой позиции → проверка exit-условий (раздел 4.4).
6. Если авто-режим ВКЛ, нет позиции, `S_entry` ≥ порога и `expected_pnl > 0`
   → `PortfolioEngine.open_position`.
7. Пересчёт equity → push через WebSocket в UI.

В ручном режиме шаг 6 заменяется действием пользователя; выходы можно делать ручными (опция).

---

## 9. Структура репозитория

```
ZCodeProject/
├── plan.md                      # этот документ
├── README.md
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── config.py
├── app/
│   ├── main.py
│   ├── routes/
│   ├── auth.py
│   ├── ws.py
│   ├── templates/ (base, dashboard, coin, analytics, backtest)
│   └── static/ (css, js — Plotly/HTMX)
├── core/
│   ├── market_service.py
│   ├── data_lifecycle.py        # раздел 6
│   ├── prediction_service.py
│   ├── indicators.py
│   ├── strategy_engine.py       # математика entry/exit
│   ├── risk_manager.py          # sizing, ATR стоп
│   ├── portfolio_engine.py      # сделки, PnL, эквити
│   ├── trading_loop.py
│   └── backtester.py
├── db/
│   ├── models.py
│   └── database.py
├── kronos/                      # vendored из Kronos repo (model/)
├── tests/
│   ├── test_indicators.py
│   ├── test_strategy.py
│   ├── test_portfolio.py
│   ├── test_data_lifecycle.py
│   └── test_break_even.py
└── data/                        # SQLite, кэш прогнозов
```

---

## 10. Этапы разработки (roadmap)

1. **Скелет + зависимости**: FastAPI hello-world, Docker, requirements, структура папок.
2. **DataLifecycleManager + MarketService**: CCXT, cold/warm start, pruning (раздел 6).
3. **Indicators**: RSI/MACD/EMA/ATR/BB/ADX на pandas, unit-тесты.
4. **Kronos integration**: загрузка модели (HF), сервис прогноза с кэшем, нормализация.
5. **Strategy engine + risk manager**: математика 4.3–4.6, тесты (вкл. тест безубыточности).
6. **Portfolio engine**: журнал сделок, расчёт комиссий/slippage, equity curve, paper-баланс $100.
7. **Persistence**: SQLAlchemy-модели, миграции.
8. **Trading loop**: APScheduler, интеграция всех модулей.
9. **UI**: дашборд, экран монеты (Plotly + прогноз), аналитика, настройки (Jinja2 + HTMX).
10. **WebSocket**: live-обновления PnL/позиций.
11. **Backtester**: исторический прогон стратегии с отчётом.
12. **Тестирование**: интеграционные тесты, проверка безубыточности на истории, валидация.

---

## 11. Риски и ограничения (честно)

- **Kronos — не гарантия прибыли.** Zero-shot прогноз на крипте может ошибаться →
  ансамбль + комиссионный фильтр + risk/reward обязательны.
- **Прошлые результаты ≠ будущие.** Бэктест валидирует математику, не гарантирует live-результат.
- **Только spot long** → в медвежьем рынке стратегия будет в кэше (это защита, не баг).
- **Реальный режим не реализуется** — только архитектурная заглушка; потребуется отдельная
  работа по безопасности API-ключей, исполнению, мониторингу.
- **Прототип** — упрощённая persistence (SQLite), нет многопользовательности, упрощённая
  отказоустойчивость.
