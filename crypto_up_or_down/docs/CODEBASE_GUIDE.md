# Codebase Guide — Polymarket Trading Toolkit

A deep-dive reference for understanding every file, how they wire together,
and where to add new strategies and data sources.
Written for intermediate Python, new to UV and Nix.

---

## Table of Contents

1. [Tooling Primer (UV + Nix)](#tooling-primer-uv--nix)
2. [Repository Layout](#repository-layout)
3. [Package-by-Package Breakdown](#package-by-package-breakdown)
   - [core](#packagescorepolymarket_algocorex)
   - [data](#packagesdatapolymarket_algodatax)
   - [indicators](#packagesindicatorspolymarket_algoindicatorsx)
   - [strategies](#packagesstrategiespolymarket_algostrategiesx)
   - [backtest](#packagesbacktestpolymarket_algobacktestx)
   - [executor](#packagesexecutorpolymarket_algoexecutorx)
4. [Scripts](#scripts)
5. [Legacy Files](#legacy-files)
6. [Config & Environment](#config--environment)
7. [Tests](#tests)
8. [How Everything Works Together](#how-everything-works-together)
9. [Flow Diagrams](#flow-diagrams)
10. [Adding a New Strategy](#adding-a-new-strategy)
11. [Adding a New Data Source](#adding-a-new-data-source)

---

## Tooling Primer (UV + Nix)

### UV — Python Package Manager

UV is a fast, modern Python package/project manager that replaces `pip`, `virtualenv`,
and `pyenv` in one tool. It reads `pyproject.toml` files.

```
# Analogy: UV is to Python what npm is to Node.js
uv sync --all-packages   →  like "npm install"
uv run python foo.py     →  like "npm run foo" (auto-activates the venv)
uv add requests          →  like "npm install requests"
uv.lock                  →  like "package-lock.json" — exact pinned versions
```

**Workspace** is the key concept here. The root `pyproject.toml` declares a workspace
with multiple members (packages). All packages share a single `.venv/` and a single
`uv.lock`, but each has its own `pyproject.toml` and is independently importable.
This is why you can `from polymarket_algo.core import Strategy` from any package.

```
[tool.uv.workspace]
members = ["packages/core", "packages/data", ...]
```

When you run `uv sync --all-packages`, UV installs every workspace package in
editable mode (like `pip install -e`), meaning imports resolve live to the source files —
no reinstalling needed as you edit code.

### Nix — Reproducible Dev Shell (optional)

Nix is a declarative package manager for *system-level* tools (Python itself, uv, ruff,
compilers). It guarantees that every developer gets the exact same tool versions.

```
flake.nix      ← defines what goes in the dev shell
flake.lock     ← pins exact nixpkgs commit hash (like uv.lock but for system tools)
nix develop    ← drops you into the configured shell
```

**You do NOT need Nix.** It is optional. If you have Python 3.13 and uv installed
through any means (system package manager, homebrew, etc.), just run
`uv sync --all-packages` and you're set.

The `flake.nix` also auto-runs `uv sync` and `prek install` (git hooks) on shell entry,
so Nix users get a zero-effort setup.

---

## Repository Layout

```
polymarket_auto_trader/
│
├── pyproject.toml          ← root workspace config + dev tools (ruff, pytest, ty)
├── uv.lock                 ← pinned dependency lockfile
├── flake.nix               ← Nix dev shell definition
├── .env.example            ← template for your secrets/config
├── process-compose.yaml    ← optional: run multiple services together
│
├── packages/               ← all installable, importable packages
│   ├── core/               ← Protocol types, config, plugin registry
│   ├── data/               ← Binance OHLCV fetcher + storage
│   ├── indicators/         ← EMA, SMA, RSI, MACD, Bollinger Bands
│   ├── strategies/         ← Streak, CopyTrade, CandleDirection, SelectiveFilter
│   ├── backtest/           ← Engine, parameter sweep, walk-forward, metrics
│   └── executor/           ← Polymarket client, WebSocket, trader, blockchain
│
├── scripts/                ← CLI entry points (uv run python scripts/X.py)
│   ├── bot.py              ← wrapper → legacy bot.py main()
│   ├── backtest.py         ← run backtest on candle data
│   ├── fetch_data.py       ← download Binance OHLCV data
│   ├── copybot.py          ← copy-trade bot entry point
│   ├── history.py          ← view/export trade history
│   └── run_backtests.py    ← batch backtests across multiple strategies
│
├── examples/
│   └── custom_strategy/    ← shows how to write a plugin strategy
│
├── tests/                  ← pytest test suite
│
├── docs/                   ← architecture and convention docs
│
├── data/                   ← OHLCV CSV/Parquet files (gitignored except samples)
├── backtest_results/        ← output files from backtests (gitignored)
│
│   # Legacy files (kept for backward compatibility, not the canonical code)
├── bot.py                  ← original streak bot
├── copybot.py              ← original copytrade bot
├── copybot_v2.py           ← improved copytrade bot
├── backtest_engine.py      ← original backtest engine
├── src/                    ← original source before monorepo migration
├── indicators/             ← original indicators (pre-migration copies)
└── strategies/             ← original strategies (pre-migration copies)
```

---

## Package-by-Package Breakdown

### `packages/core/polymarket_algo/core/`

The foundation everything else depends on. Zero external state — just type definitions
and configuration.

---

#### `types.py`

Defines the three **Protocols** that everything plugs into. A Python `Protocol` is like
an interface: any class that has the right methods/attributes satisfies it, without
needing to inherit from anything.

```python
class Indicator(Protocol):
    name: str
    def compute(self, series: pd.Series, **params) -> pd.Series | pd.DataFrame: ...
```

**`Indicator`** — any class with a `name` string and a `compute()` method that takes
a pandas Series of prices and returns a Series or DataFrame.

```python
class Strategy(Protocol):
    name: str
    description: str
    timeframe: str
    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame: ...
    @property
    def default_params(self) -> dict: ...
    @property
    def param_grid(self) -> dict[str, list]: ...
```

**`Strategy`** — the core contract every strategy must satisfy.
- `evaluate()` takes a OHLCV DataFrame and returns a DataFrame with at minimum
  a `signal` column (integers: `1`=long/up, `-1`=short/down, `0`=no trade) and
  optionally a `size` column (USD amount to bet).
- `default_params` — the baseline parameter values used when no overrides provided.
- `param_grid` — the search space for parameter optimization (used by `parameter_sweep`).

```python
@dataclass
class PriceTick:
    symbol: str
    price: float
    timestamp: float
    size: float | None = None
    side: str | None = None   # "buy" | "sell"
    source: str = ""
```

**`PriceTick`** — the normalized event object emitted by every data feed. No matter
whether the price came from Polymarket WebSocket, Binance REST, or a future Chainlink
feed, it becomes a `PriceTick` before anything acts on it.

```python
@runtime_checkable
class DataFeed(Protocol):
    name: str
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def subscribe(self, symbol: str, **kwargs) -> None: ...
    def unsubscribe(self, symbol: str) -> None: ...
    def on_tick(self, callback: Callable[[PriceTick], None]) -> None: ...
    def on_reconnect(self, callback: Callable[[], None]) -> None: ...
    def is_connected(self) -> bool: ...
```

**`DataFeed`** — the interface any real-time data source must implement. The
`@runtime_checkable` decorator means you can use `isinstance(feed, DataFeed)` at
runtime, which is rare for Protocols. `on_tick` registers a callback that fires every
time a new price arrives.

---

#### `config.py`

A single `Config` class (not instantiated — everything is a class attribute) that reads
from environment variables via `python-dotenv`. It calls `load_dotenv()` at import time,
which reads your `.env` file and populates `os.environ`.

Key configuration groups:
- **Wallet**: `PRIVATE_KEY` — your Polygon EOA private key for live trading
- **APIs**: hardcoded Polymarket URLs (Gamma API, CLOB API, WebSocket URLs)
- **Strategy**: `STREAK_TRIGGER`, `BET_AMOUNT`, `MIN_BET`, `MAX_DAILY_BETS`, `MAX_DAILY_LOSS`
- **Timing**: `ENTRY_SECONDS_BEFORE` — how many seconds before a window to place the bet
- **Mode**: `PAPER_TRADE=true` by default — forces simulation, never defaults to live
- **Resilience**: Circuit breaker threshold, rate limit, REST timeout/retries
- **WebSocket**: URLs and `USE_WEBSOCKET` toggle
- **Selective filter**: thresholds for the copytrade quality filter
- **Delay model**: coefficients for the copy-trade price impact calculator

Also creates `LOCAL_TZ` — a timezone object built from `TIMEZONE_NAME` env var.
Used throughout for display timestamps.

---

#### `plugin.py`

Implements the **plugin discovery system** — how the toolkit finds strategies and
indicators at runtime without hardcoding imports.

There are three discovery mechanisms:

1. **Entry points** (`_discover()`): Python's standard plugin system. When you install
   a package (like `example-polymarket-strategy`), its `pyproject.toml` can declare:
   ```toml
   [project.entry-points."polymarket_algo.strategies"]
   rsi_reversal = "example_strategy.rsi_reversal:RSIReversalStrategy"
   ```
   `importlib.metadata.entry_points()` reads these declarations from all installed
   packages and loads the class. This is how `examples/custom_strategy` works.

2. **Local file plugins** (`load_local_plugins()`): Scans `~/.polymarket-algo/plugins/*.py`
   and dynamically loads any class that has an `evaluate()` (Strategy) or `compute()`
   (Indicator) method. Drop a Python file there and it's auto-discovered — no install needed.

3. **`PluginRegistry`**: Combines both, stores results in `.strategies` and `.indicators`
   dicts mapping name → class. Call `PluginRegistry().load()` to populate it.

---

### `packages/data/polymarket_algo/data/`

> Note: The `data` package source wasn't in the file listing shown to me, but `scripts/fetch_data.py`
> imports `from polymarket_algo.data.binance import INTERVALS, START, SYMBOLS, fetch_klines`.

The data package wraps the **Binance public API** to download historical OHLCV
(Open/High/Low/Close/Volume) candlestick data.

- `fetch_klines(symbol, interval, start_ms, end_ms)` — paginates through the Binance
  `/api/v3/klines` endpoint and returns a pandas DataFrame with columns:
  `open_time`, `open`, `high`, `low`, `close`, `volume`
- `SYMBOLS` — list of trading pairs to fetch (e.g., `["BTCUSDT", "ETHUSDT"]`)
- `INTERVALS` — list of timeframe strings (e.g., `["1h", "4h"]`)
- `START` — the historical start date for fetching

Data is saved to `data/` as Parquet files: `data/btc_1h.parquet`, `data/eth_4h.parquet`, etc.
Parquet is a columnar binary format — much faster and smaller than CSV for numerical data.

---

### `packages/indicators/polymarket_algo/indicators/`

Pure mathematical functions on pandas Series. No state, no side effects.
Each file exports a single function or class.

---

#### `ema.py`

```python
def ema(series: pd.Series, period: int = 20) -> pd.Series:
```

**Exponential Moving Average** — weights recent prices more heavily than older ones.
Uses `pandas.ewm(span=period)`. The `span` parameter relates to the decay factor:
alpha = 2/(span+1). With `min_periods=period`, returns `NaN` for the first `period-1` rows
(not enough data to compute a valid EMA).

---

#### `sma.py`

**Simple Moving Average** — plain rolling mean over the last `period` candles.
Equal weighting on all prices in the window.

---

#### `rsi.py`

```python
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
```

**Relative Strength Index** — measures momentum. Returns values 0–100.
Above 70 = overbought (price ran up fast), below 30 = oversold (price dropped fast).

Implementation uses Wilder's smoothing (EWM with alpha=1/period):
1. Compute price changes (`.diff()`)
2. Separate gains (positive changes) from losses (negative changes)
3. Apply EWM smoothing to get `avg_gain` and `avg_loss`
4. `RSI = 100 - (100 / (1 + avg_gain/avg_loss))`

Special cases: both flat → RSI=50, avg_loss=0 → RSI=100.

---

#### `macd.py`

```python
def macd(series, fast_period=12, slow_period=26, signal_period=9) -> pd.DataFrame:
```

**Moving Average Convergence/Divergence** — trend-following momentum indicator.
Returns a DataFrame with three columns:
- `macd` — the difference between fast EMA and slow EMA (the "MACD line")
- `signal` — EMA of the MACD line (the "signal line")
- `histogram` — MACD minus signal (shows momentum direction and strength)

A positive histogram means momentum is growing bullish. Crossing from negative to
positive is a traditional buy signal.

---

#### `bollinger.py`

**Bollinger Bands** — volatility bands around a moving average.
Returns a DataFrame with `upper`, `middle`, `lower` columns.
- Middle = SMA
- Upper = SMA + (N * standard deviation)
- Lower = SMA - (N * standard deviation)

Price touching the upper band = potentially overbought, lower = potentially oversold.

---

#### `__init__.py`

Re-exports all indicator functions at the package level so you can write:
```python
from polymarket_algo.indicators import ema, rsi, macd
```
instead of importing from submodules.

---

### `packages/strategies/polymarket_algo/strategies/`

Each file contains one strategy class satisfying the `Strategy` Protocol.
All `evaluate()` methods receive a DataFrame with OHLCV columns and return a DataFrame
with `signal` (int) and `size` (float) columns indexed the same as `candles`.

---

#### `streak_reversal.py` — `StreakReversalStrategy`

**Concept**: Markets sometimes trend in one direction for several consecutive candles.
This strategy bets that a streak will reverse.

**Logic**:
1. Compute direction for each candle: `close > prev_close → 1`, else `→ -1`
2. Count consecutive same-direction candles using groupby trick:
   - `(direction != direction.shift()).cumsum()` creates a group ID that increments
     each time direction changes
   - `.cumcount()` within each group gives the streak length
3. When streak length ≥ `trigger` (default 4):
   - If the streak is up: signal = `-1` (bet DOWN, expecting reversal)
   - If the streak is down: signal = `1` (bet UP, expecting reversal)

**Parameters** (for backtest optimization):
- `trigger`: minimum streak length (tested: 3, 4, 5)
- `size`: bet amount in USD (tested: $10, $15, $20)

---

#### `candle_direction.py` — `CandleDirectionStrategy`

**Concept**: Only trade when EMA trend, MACD momentum, and RSI all agree.
A multi-indicator confluence approach — fewer signals but higher quality.

**Logic**:
1. Fast EMA vs Slow EMA: if fast > slow → bullish trend (`bullish_ema`)
2. MACD line vs Signal line: if MACD > signal → bullish momentum (`bullish_macd`)
3. RSI in normal range (not overbought): `rsi > oversold` → conditions allow a long
4. **Long signal** (`signal=1`): all three are bullish
5. **Short signal** (`signal=-1`): all three are bearish
6. **Strong signal** (increases `size` to $20): bullish MACD histogram + RSI 50–65 range

**Parameters**: ema_fast, ema_slow, rsi_period, rsi_overbought, rsi_oversold,
macd_fast, macd_slow, macd_signal — all swept during optimization.

---

#### `copytrade.py` — `CopytradeStrategy`

A stub/marker class. Copytrade is **event-driven** (reacts to another wallet's trades),
not candle-driven. Calling `evaluate()` raises `NotImplementedError`.

Its purpose is to be discovered by the plugin registry and identify itself in
the system, not to produce signals from historical data.
The actual copytrade logic lives in `scripts/copybot.py` and the legacy `copybot*.py` files.

---

#### `selective_filter.py` — `SelectiveFilter`

Not a strategy itself — a **pre-trade quality gate** used by the copytrade system.
Before copying a trade, the bot calls `should_trade(signal, market, execution_info)`.

Checks (all configurable via `.env`):
1. **Delay** — if the copy lag > `SELECTIVE_MAX_DELAY_MS` (default 20s), skip
2. **Fill price** — must be between `SELECTIVE_MIN_FILL_PRICE` (0.55) and `SELECTIVE_MAX_FILL_PRICE` (0.80)
   — avoids highly lopsided markets where probability is already priced in
3. **Price movement** — if price moved > `SELECTIVE_MAX_PRICE_MOVEMENT_PCT` (15%) since signal, skip
4. **Spread** — if bid-ask spread > `SELECTIVE_MAX_SPREAD` (0.025), market is illiquid, skip
5. **Volatility factor** — derived from the delay model's spread ratio calculation
6. **Depth** — minimum liquidity at best price level

Returns `(True, "all checks OK")` or `(False, "reason string")`.

---

### `packages/backtest/polymarket_algo/backtest/`

The simulation engine. Runs strategies against historical data to measure performance.

---

#### `engine.py`

The core workhorse. Three public functions:

**`run_backtest(candles, strategy, strategy_params, buy_price, win_payout)`**

This is the simplest simulation model: binary outcome, fixed payoff.
Polymarket BTC 5-min markets work like this — you buy a token at some price,
and it pays $1 if BTC went up (or down), else $0.

Step by step:
1. Call `strategy.evaluate(candles)` → get `signal` and `size` columns
2. Compute `outcome_up`: was the next candle's close higher than current? (0 or 1)
3. `active` rows: where signal ≠ 0 AND next candle exists
4. Determine `wins`: signal=1 and price went up, OR signal=-1 and price went down
5. Compute `per_share_pnl`: `+win_payout - buy_price` on wins, `-buy_price` on losses
6. `trade_pnl = per_share_pnl * size` (scaled by position size)
7. Cumsum gives the equity curve

Metrics computed:
- `win_rate` — fraction of trades that won
- `total_pnl` — net profit/loss in USD
- `max_drawdown` — largest peak-to-trough drop in the equity curve
- `sharpe_ratio` — (mean return / std deviation) * sqrt(n) — risk-adjusted return
- `trade_count` — how many trades were taken

Returns a `BacktestResult(metrics, trades, pnl_curve)`.

**`parameter_sweep(candles, strategy, param_grid)`**

Tries every combination in `param_grid` (Cartesian product via `itertools.product`).
For each combo, runs `run_backtest` and collects metrics.
Returns a DataFrame sorted by `win_rate` then `total_pnl`, best first.

Useful for finding: "which `trigger` value and `size` produced the best win rate
on historical data?"

**`walk_forward_split(candles, train_ratio=0.75)`**

Splits data into train (first 75%) and test (last 25%) sets.

The workflow: optimize parameters on `train`, then verify with `run_backtest` on `test`.
This avoids overfitting — you don't want parameters tuned to the test data.

---

#### `metrics.py`

Single exported function `max_drawdown(equity_curve)` — calculates the worst
peak-to-trough loss. Also used internally by `engine.py`.

---

#### `report.py`

Single exported function `format_metrics(metrics)` — returns a one-liner string
summary like `win_rate=52.30% pnl=128.50 trades=147`. Used for quick console output.

---

### `packages/executor/polymarket_algo/executor/`

The live trading layer. Handles all external API calls, order placement, and real-time
data streaming.

---

#### `client.py` — `PolymarketClient`, `Market`, `DelayImpactModel`

**`Market` dataclass** — represents a single BTC 5-min prediction market:
- `timestamp`: the Unix timestamp that identifies the market window (e.g., 1771051500)
- `slug`: human-readable ID like `"btc-updown-5m-1771051500"`
- `up_token_id` / `down_token_id`: CLOB token IDs needed to place orders
- `up_price` / `down_price`: current probability (0–1) for each outcome
- `accepting_orders`: whether the market window is still open for trading
- `taker_fee_bps`: fee rate in basis points (e.g., 1000 = 10% base rate)
- `resolved`: whether the UMA oracle has confirmed the final outcome

**`PolymarketClient`** — read-only REST client for market data.
Uses `requests.Session` with connection pooling (20 connections) and retry logic.
Has a two-level cache: token IDs (never change) and market data (TTL-based).

Key methods:
- `get_market(timestamp)` — fetches from Gamma API → parses event/market JSON
- `get_recent_outcomes(count=10)` — walks backwards through resolved windows
- `get_next_market_timestamp()` — calculates the next 5-min boundary
- `get_orderbook(token_id)` — fetches the full order book from CLOB API
- `get_price(token_id, side)` — fastest single-price fetch
- `get_spread(token_id)` → (best_bid, best_ask)
- `get_execution_price(token_id, side, amount_usd, copy_delay_ms)` — simulates
  walking the order book to compute realistic fill price + slippage + delay impact

**Fee formula**: `fee = price * (1 - price) * base_fee_bps / 10000`
At 50¢ with default 10% base: `0.5 * 0.5 * 0.10 = 2.5%` effective fee rate.

**`DelayImpactModel`** — models the price impact of copying a trade with delay.
Used by copytrade to estimate how much worse your fill price is vs the trader you're copying.

Formula: `impact = sqrt(delay_seconds) * base_coef * liquidity_factor * volatility_factor`
- sqrt decay: impact grows fast at first, then slower (1s → 0.8%, 4s → 1.6%, 9s → 2.4%)
- Liquidity factor: larger order relative to depth-at-best → more impact
- Volatility factor: wider spread → more volatile market → more impact

---

#### `trader.py` — `Trade`, `TradingState`, `PaperTrader`, `LiveTrader`

**`Trade` dataclass** — the complete record of a single bet. Has ~45 fields organized
into logical groups: core fields, resolution fields, settlement breakdown, copytrade fields,
simulation fields, price movement, market context, pattern analysis (time-of-day, session
tracking, streak tracking), on-chain data. This is your analytics database.

Serialization:
- `to_nested_json()` — saves to a structured dict with logical groupings (market, position,
  execution, fees, copytrade, settlement, context, session, timing, on_chain)
- `from_nested_json()` — reconstructs a Trade from that dict
- `to_history_dict()` — flatter format for human-readable export

**`TradingState`** — persistent session state that survives bot restarts.
- Holds a list of `Trade` objects (working set: last 100)
- Tracks `daily_bets`, `daily_pnl`, `bankroll`
- `save()` → writes `trades.json` (working state) + appends to `trade_history_full.json`
- `load()` → reads from `trades.json`, populates `_saved_trade_ids` from full history
- `can_trade()` → checks daily bet limit, daily loss limit, minimum bankroll
- `settle_trade()` → calculates gross payout, fee, net profit when market resolves
- `backfill_settlements()` → queries Polymarket API to settle trades that were
  pending when the bot was restarted
- `update_unrealized_pnl()` → for pending trades, computes expected value based
  on current market prices

**`PaperTrader`** — simulates trade execution without touching real money.
- Queries the real order book (or uses WebSocket cache) to compute realistic fill price
- Applies slippage by walking the book
- Applies the fee model
- Applies copy delay impact for copytrade signals
- Records all this in the `Trade` dataclass for analysis
- Returns a `Trade` object with `paper=True`

**`LiveTrader`** — places real orders via `py-clob-client`.
- Requires `PRIVATE_KEY` in environment
- Initializes `ClobClient` with wallet credentials and derives API keys
- Places **FOK (Fill-Or-Kill)** market orders — fills immediately at best price or
  is cancelled, no partial resting on the book
- Polls order status after submission to confirm fill
- Returns a `Trade` object with `paper=False` and real `order_id`

---

#### `ws.py` — `PolymarketWebSocket`, `UserWebSocket`, `MarketDataCache`

**`PolymarketWebSocket`** — real-time market data via WebSocket.
Gives ~100ms latency vs ~1s for REST polling. Critical for copytrade.

Architecture:
- Runs an `asyncio` event loop in a **background daemon thread** (so the main bot loop
  is synchronous and unblocked)
- Maintains `_orderbooks` dict: `token_id → CachedOrderBook`
- Message types handled:
  - `"book"` → full snapshot → replaces entire order book
  - `"price_change"` → delta → applies incremental updates to specific price levels
  - `"last_trade_price"` → trade event → fires `on_trade` callback

Reconnection: exponential backoff (2^n seconds, capped at 30s).
Re-subscribes to all markets automatically after reconnect.

**`CachedOrderBook`** — in-memory order book state.
- `bids` and `asks` sorted lists of `OrderBookLevel(price, size)`
- `get_execution_price(side, amount_usd)` — walks the book to simulate fill

**`UserWebSocket`** — authenticated WebSocket for live trading order updates.
Connects to the user channel and sends API key/secret/passphrase for auth.
Fires `on_order_update` callback when an order changes status
(MATCHED → filled, MINED, CONFIRMED, FAILED).

**`MarketDataCache`** — high-level cache that combines WebSocket + REST fallback.
- `get_orderbook()` — tries WebSocket cache first (must be < 5s old), falls back to REST
- `get_execution_price()` — tries WebSocket cache (< 2s old for execution), falls back to REST
- `on_trade()` — register callbacks for trade events
- `prefetch_markets()` — pre-fetches token IDs and subscribes to WebSocket for upcoming
  market windows

This is the class injected into `PaperTrader` and `LiveTrader` to speed up orderbook
lookups vs making individual REST calls for each trade.

---

#### `feed.py` — `PolymarketDataFeed`

A thin adapter that wraps `PolymarketWebSocket` and exposes the `DataFeed` Protocol.

Why does this exist separately from `ws.py`? To decouple the Protocol interface
(what the rest of the system sees) from the implementation details (WebSocket specifics).
If you wanted to swap in a different Polymarket data source, you'd only need to create
a new class that satisfies `DataFeed` — the rest of the system is unchanged.

- Trade events from WebSocket → converted to `PriceTick` → fired to `on_tick` callbacks
- Mid-price changes from orderbook → also converted to `PriceTick` with `source="polymarket-mid"`

---

#### `resilience.py` — `CircuitBreaker`, `RateLimiter`, `HealthCheck`, `with_retry`

Production-grade failure handling patterns.

**`CircuitBreaker`** — prevents cascading failures when an API is down.
Three states:
- `CLOSED` (normal): all requests pass through
- `OPEN` (failing): all requests blocked immediately (don't waste time trying)
- `HALF_OPEN` (recovering): allows a few test requests; if they succeed → CLOSED,
  if any fail → back to OPEN

Transitions:
- CLOSED → OPEN: when `_failures >= failure_threshold` (default 5)
- OPEN → HALF_OPEN: automatically after `recovery_time` seconds (default 60s)
- HALF_OPEN → CLOSED: after 3 consecutive successes
- HALF_OPEN → OPEN: on any failure

**`RateLimiter`** — sliding window algorithm.
Tracks request timestamps in a `deque`. Before each request, evicts timestamps outside
the 60s window, then checks if count < limit (default 120/min).
If over limit, caller can wait `time_until_allowed()` seconds.

**`with_retry(fn, max_retries, base_delay, max_delay, circuit_breaker, rate_limiter)`** —
wraps any function call with:
- Rate limiter check (waits if needed)
- Circuit breaker check (raises immediately if open)
- Exponential backoff between retries (1s, 2s, 4s, 8s, ...)
- Smart retry logic: FATAL errors (401, 403, 404, invalid params) → don't retry

**`categorize_error(error)`** — classifies any exception:
- 429/rate limit → `RATE_LIMITED`
- 5xx/timeout/connection → `RETRYABLE`
- 4xx (non-429) / "invalid" / "insufficient" → `FATAL`
- Unknown → `RETRYABLE` (conservative default)

**`HealthCheck`** — register check functions by name, query all health statuses.

---

#### `blockchain.py` — `PolygonscanClient`

Queries Polygonscan (Etherscan v2 API with `chainid=137` for Polygon mainnet)
to fetch on-chain transaction data for live orders.

Given a transaction hash from a filled order, retrieves:
- Block number and timestamp
- Gas limit and actual gas used
- Gas price in gwei
- Transaction fee in MATIC

Stored in the `Trade` dataclass for cost analysis (how much did gas cost on each trade?).
Only used in live trading mode; requires a `POLYGONSCAN_API_KEY` in `.env`.

---

## Scripts

### `scripts/bot.py`

Thin wrapper that imports and calls `main()` from the legacy `bot.py` at the project root.
The actual bot logic lives in `bot.py` (root) which uses `src/` imports.

```
uv run python scripts/bot.py --paper
```

**What the bot does** (loop every 1 second):
1. **Settle pending trades** — check if any pending markets have resolved (closed + outcome known)
2. **Check trading limits** — can_trade() verifies daily bet count, daily loss, bankroll
3. **Calculate timing** — figure out which 5-min window to target (always the next one)
4. **Wait for entry window** — sleep until we're within `ENTRY_SECONDS_BEFORE` seconds of target
5. **Fetch recent outcomes** — `get_recent_outcomes(trigger + 2)` gets N resolved windows
6. **Evaluate strategy** — call streak `evaluate()` to see if there's a signal
7. **Get market data** — fetch the target market's token IDs and prices
8. **Size the bet** — Kelly criterion + cap at `BET_AMOUNT` and 10% of bankroll
9. **Place bet** — PaperTrader or LiveTrader
10. **Record and save** — append to state, save JSON

### `scripts/backtest.py`

Loads BTC 1h Parquet data, splits into train/test, runs `parameter_sweep` on train,
picks the best parameters, validates with `run_backtest` on test, prints metrics.

### `scripts/fetch_data.py`

Downloads Binance OHLCV data for configured symbols and intervals.
Saves Parquet files to `data/`. Run this before backtesting if you don't have data files.

### `scripts/run_backtests.py`

Batch backtester — runs multiple strategy/parameter combinations and writes results
to `backtest_results/`.

### `scripts/history.py`

CLI tool for viewing and exporting trade history. Reads `trade_history_full.json`.
Supports `--stats` (show aggregate statistics) and `--export csv` / `--export json`.

---

## Legacy Files

These exist at the project root for backward compatibility. They use imports from `src/`
(the pre-monorepo code). They are **not the canonical code** — the `packages/` versions
are the canonical versions.

| File | What it is |
|------|------------|
| `bot.py` | Original streak bot (uses `src/strategies/streak.py`) |
| `copybot.py` | Original copytrade bot (REST polling) |
| `copybot_v2.py` | Improved copytrade bot (WebSocket-backed) |
| `backtest_engine.py` | Original backtest engine |
| `src/` | Pre-migration source tree — contains `core/`, `strategies/`, `infra/` |
| `indicators/` | Pre-migration indicator copies |
| `strategies/` | Pre-migration strategy copies |

The `src/` directory mirrors the `packages/executor/` structure but with older code.
Over time these will be removed; when modifying or adding features, always work in
`packages/`, not `src/`.

---

## Config & Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
# Required for live trading only:
PRIVATE_KEY=0x...

# Strategy tuning:
STREAK_TRIGGER=4        # detect streak of N same outcomes
BET_AMOUNT=5            # USD per bet
MIN_BET=1               # minimum bet (exchange minimum)
MAX_DAILY_BETS=50       # hard stop after this many bets
MAX_DAILY_LOSS=50       # hard stop if PnL falls below -$50 today

# Mode:
PAPER_TRADE=true        # ALWAYS true until you're ready to risk real money

# Timing:
ENTRY_SECONDS_BEFORE=30 # enter 30s before the 5-min window starts
TIMEZONE=UTC            # for display only

# Copytrade:
COPY_WALLETS=0xabc...,0xdef...   # comma-separated wallet addresses to copy
COPY_POLL_INTERVAL=5              # seconds between polling (REST fallback)

# Selective filter (copytrade quality gate):
SELECTIVE_FILTER=false
SELECTIVE_MAX_DELAY_MS=20000
SELECTIVE_MIN_FILL_PRICE=0.55
SELECTIVE_MAX_FILL_PRICE=0.80

# Advanced:
POLYGONSCAN_API_KEY=...   # optional, for on-chain gas cost analysis
LOG_LEVEL=INFO
```

---

## Tests

`tests/` contains four test files:

| File | What it tests |
|------|---------------|
| `test_indicators.py` | EMA, RSI, MACD, Bollinger produce correct values |
| `test_backtest.py` | `run_backtest`, `parameter_sweep`, `walk_forward_split` |
| `test_strategy_protocol.py` | All strategies satisfy the Strategy Protocol |
| `test_executor_imports.py` | executor package imports without error |

Run with:
```bash
uv run pytest -v
```

---

## How Everything Works Together

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION                            │
│  .env → Config class → all packages read from Config            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                               │
│                                                                 │
│  Binance REST API ──→ fetch_klines() ──→ Parquet files          │
│         (historical OHLCV for backtesting)                      │
│                                                                 │
│  Polymarket REST API ──→ PolymarketClient                       │
│         (market info, outcomes, order books)                    │
│                                                                 │
│  Polymarket WebSocket ──→ PolymarketWebSocket                   │
│         (real-time order book deltas, trade events)             │
│                │                                                │
│                └──→ PolymarketDataFeed (DataFeed Protocol)      │
└─────────────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────┐            ┌──────────────────┐
│  INDICATORS │            │  STRATEGY ENGINE │
│             │            │                  │
│  ema()      │──────────→ │  evaluate()      │
│  rsi()      │            │   returns:       │
│  macd()     │            │   signal: int    │
│  bollinger()│            │   size:  float   │
└─────────────┘            └──────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION LAYER                               │
│                                                                 │
│  Signal = 1 (long/up) or -1 (short/down) or 0 (no trade)      │
│                                                                 │
│  [Optional] SelectiveFilter.should_trade() for copytrade        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                              │
│                                                                 │
│  PaperTrader                    LiveTrader                      │
│  ─────────                      ──────────                      │
│  Query orderbook                Same validation                 │
│  Compute execution price        Sign order (private key)        │
│  Apply slippage                 Submit FOK market order         │
│  Apply fees                     Poll order status              │
│  Record Trade(paper=True)       Record Trade(paper=False)       │
│                                                                 │
│  Both backed by MarketDataCache (WebSocket + REST fallback)     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STATE MANAGEMENT                             │
│                                                                 │
│  TradingState                                                   │
│  ────────────                                                   │
│  trades[]          ← appended on each bet                       │
│  bankroll          ← adjusted on settlement                     │
│  daily_bets        ← incremented on each bet                    │
│  daily_pnl         ← adjusted on settlement                     │
│                                                                 │
│  Persistence:                                                   │
│  trades.json           ← working state (last 100 trades)        │
│  trade_history_full.json ← full append-only history             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SETTLEMENT LOOP                               │
│                                                                 │
│  Every bot tick, check pending trades:                          │
│  get_market(ts) → if market.closed and market.outcome:          │
│    settle_trade(trade, outcome)                                 │
│    → compute shares, gross_profit, fee_amount, net_profit       │
│    → update bankroll                                            │
│    → update trade_history_full.json                             │
└─────────────────────────────────────────────────────────────────┘
```

### Backtesting vs Live Bot — Same Strategy, Different Execution

The `Strategy.evaluate()` method is **shared** between backtesting and live trading.
The difference is only in what feeds it candles:

```
BACKTESTING:
  Parquet file → DataFrame → strategy.evaluate() → signals → run_backtest() → metrics

LIVE BOT:
  Polymarket API → list of outcomes ["up","up","down","up"] → converted to candle-like
  DataFrame → strategy.evaluate() → signal → place_bet()
```

This is why backtesting on historical BTC prices is a proxy for live Polymarket
performance — the markets resolve based on BTC 5-min direction.

### Two-Phase Market Lifecycle

Every BTC 5-min market has a lifecycle:

```
T=0:00   Window opens — token IDs created, accepting_orders=True
T=0:00   to T=4:30  — Main trading period
T=4:30   Bot places bet (ENTRY_SECONDS_BEFORE=30)
T=5:00   Window closes — no new orders accepted
T=5:xx   UMA oracle resolves: up_price → 1.0 or down_price → 1.0
T=5:xx   Bot detects closed+resolved, calls settle_trade()
```

The bot always targets the **next** window, placing bets ~30 seconds before it opens.

---

## Flow Diagrams

### Streak Bot Main Loop

```
START
  │
  ▼
Load TradingState from trades.json
  │
  ▼
┌──────────────────────────────────────────┐
│              MAIN LOOP (every 1s)        │
│                                          │
│  ┌── Settle pending trades ──────────┐   │
│  │  for each pending trade:          │   │
│  │    get_market(trade.timestamp)    │   │
│  │    if closed AND outcome known:   │   │
│  │      settle_trade() + save()      │   │
│  └───────────────────────────────────┘   │
│                                          │
│  can_trade()?  ─── NO ──→ sleep(10)      │
│      │                                   │
│     YES                                  │
│      │                                   │
│  Already bet this window? ─ YES ─→ skip  │
│      │                                   │
│     NO                                   │
│      │                                   │
│  T-30s until next window? ─ NO ─→ wait   │
│      │                                   │
│     YES                                  │
│      │                                   │
│  get_recent_outcomes(trigger+2)          │
│      │                                   │
│  streak_evaluate(outcomes)               │
│      │                                   │
│  signal? ─── NO ──→ mark skipped         │
│      │                                   │
│     YES                                  │
│      │                                   │
│  get_market(next_window_ts)              │
│      │                                   │
│  Kelly size + BET_AMOUNT cap             │
│      │                                   │
│  trader.place_bet() ──→ Trade            │
│      │                                   │
│  state.record_trade(trade)               │
│  state.save()                            │
│      │                                   │
│  sleep(5)                                │
└──────────────────────────────────────────┘
  │
 SIGINT
  │
  ▼
state.save()
EXIT
```

### Backtest Flow

```
fetch_data.py
  │ download Binance OHLCV
  ▼
data/btc_1h.parquet

scripts/backtest.py
  │
  ▼
pd.read_parquet("data/btc_1h.parquet")
  │
  ▼
walk_forward_split(candles, train_ratio=0.75)
  ├─→ train (first 75%)
  └─→ test  (last 25%)
  │
  ▼
parameter_sweep(train, strategy.evaluate, PARAM_GRID)
  │ for each parameter combo:
  │   run_backtest(train, strategy, params)
  │   collect metrics
  ▼
sweep DataFrame (sorted by win_rate, total_pnl)
  │
  ▼
best_params = sweep.iloc[0]
  │
  ▼
run_backtest(test, strategy.evaluate, best_params)
  │
  ▼
BacktestResult(metrics, trades, pnl_curve)
  │
  ▼
print metrics
```

### WebSocket Data Flow

```
Polymarket WSS Server
       │
       │  JSON messages:
       │  • "book" (full snapshot)
       │  • "price_change" (delta)
       │  • "last_trade_price" (trade)
       │
       ▼
PolymarketWebSocket._handle_message()
       │
       ├── "book" ─────────────→ CachedOrderBook.update_from_snapshot()
       │                              └─→ bids[], asks[], best_bid, best_ask, mid
       │
       ├── "price_change" ─────→ CachedOrderBook.update_from_delta()
       │                              └─→ update single price level
       │
       └── "last_trade_price" ─→ TradeEvent
                                      │
                        ┌────────────┘
                        │
              on_trade callback
                        │
           ┌────────────┴────────────────┐
           │                             │
    MarketDataCache              PolymarketDataFeed
    ._handle_trade()              ._handle_trade()
           │                             │
    dispatch to                  convert to PriceTick
    registered callbacks         emit to on_tick callbacks
```

### Plugin Discovery Flow

```
PluginRegistry.load()
       │
       ├── entry_points(group="polymarket_algo.strategies")
       │       │
       │       │  reads pyproject.toml from ALL installed packages
       │       ▼
       │   [StreakReversalStrategy, CandleDirectionStrategy,
       │    CopytradeStrategy, RSIReversalStrategy, ...]
       │
       ├── entry_points(group="polymarket_algo.indicators")
       │       ▼
       │   [EMAIndicator, RSIIndicator, ...]
       │
       └── load_local_plugins()  (~/.polymarket-algo/plugins/*.py)
               │
               │  dynamically imports each .py file
               │  finds classes with evaluate() or compute()
               ▼
           [YourCustomStrategy, ...]
```

---

## Adding a New Strategy

Here's exactly what you need to create and where:

### Option A: In the packages/strategies package (built-in)

1. Create `packages/strategies/src/polymarket_algo/strategies/my_strategy.py`:

```python
from __future__ import annotations
import pandas as pd
from typing import Any

class MyStrategy:
    name = "my_strategy"
    description = "What it does"
    timeframe = "5m"

    @property
    def default_params(self) -> dict[str, Any]:
        return {"window": 20, "threshold": 0.5, "size": 10.0}

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "window": [10, 20, 30],
            "threshold": [0.3, 0.5, 0.7],
        }

    def evaluate(self, candles: pd.DataFrame, **params: Any) -> pd.DataFrame:
        config = {**self.default_params, **params}

        close = candles["close"]
        # ... your signal logic ...
        signal = pd.Series(0, index=candles.index, dtype=int)
        # signal = 1 → bet UP, signal = -1 → bet DOWN, 0 → no trade

        size = pd.Series(config["size"], index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
```

2. Export it from `packages/strategies/src/polymarket_algo/strategies/__init__.py`

3. Register it as an entry point in `packages/strategies/pyproject.toml`:
```toml
[project.entry-points."polymarket_algo.strategies"]
my_strategy = "polymarket_algo.strategies.my_strategy:MyStrategy"
```

4. Run `uv sync --all-packages` to reinstall with the new entry point.

### Option B: As a standalone plugin (no package install needed)

Create `~/.polymarket-algo/plugins/my_strategy.py` with the same class structure.
It will be auto-discovered when `PluginRegistry().load()` is called.

### Backtest your strategy

```python
from polymarket_algo.backtest.engine import run_backtest, parameter_sweep, walk_forward_split
from my_strategy import MyStrategy
import pandas as pd

candles = pd.read_parquet("data/btc_1h.parquet")
train, test = walk_forward_split(candles)

strategy = MyStrategy()
# Find best parameters:
sweep = parameter_sweep(train, strategy.evaluate, strategy.param_grid)
print(sweep.head())

# Test best params on unseen data:
best_params = sweep.iloc[0].to_dict()
result = run_backtest(test, strategy.evaluate, best_params)
print(result.metrics)
```

---

## Adding a New Data Source

### What is a Data Source?

Anything that provides real-time price ticks. Must satisfy the `DataFeed` Protocol.

### Minimal Implementation

```python
# my_feed.py
import time
from collections.abc import Callable
from polymarket_algo.core.types import DataFeed, PriceTick


class MyCustomFeed:
    """DataFeed Protocol implementation for MyDataSource."""

    name = "my-source"

    def __init__(self):
        self._tick_callbacks: list[Callable[[PriceTick], None]] = []
        self._reconnect_callbacks: list[Callable[[], None]] = []
        self._connected = False

    def start(self) -> None:
        """Connect and begin streaming."""
        self._connected = True
        # Start your connection (WebSocket, SSE, polling thread, etc.)
        # When you get a new price, call self._emit_tick(tick)

    def stop(self) -> None:
        """Disconnect cleanly."""
        self._connected = False

    def subscribe(self, symbol: str, **kwargs) -> None:
        """Subscribe to price updates for a symbol."""
        # Tell your data source to start sending prices for `symbol`
        pass

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from a symbol."""
        pass

    def on_tick(self, callback: Callable[[PriceTick], None]) -> None:
        """Register a callback for new price ticks."""
        self._tick_callbacks.append(callback)

    def on_reconnect(self, callback: Callable[[], None]) -> None:
        """Register a callback for reconnection events."""
        self._reconnect_callbacks.append(callback)

    def is_connected(self) -> bool:
        return self._connected

    def _emit_tick(self, symbol: str, price: float) -> None:
        """Call this whenever you receive a new price."""
        tick = PriceTick(
            symbol=symbol,
            price=price,
            timestamp=time.time(),
            source=self.name,
        )
        for cb in self._tick_callbacks:
            cb(tick)
```

### Verify it satisfies the Protocol

```python
from polymarket_algo.core.types import DataFeed

feed = MyCustomFeed()
assert isinstance(feed, DataFeed)  # works because DataFeed is @runtime_checkable
```

### Use it alongside the existing Polymarket feed

```python
from polymarket_algo.executor.feed import PolymarketDataFeed
from my_feed import MyCustomFeed

poly_feed = PolymarketDataFeed()
custom_feed = MyCustomFeed()

# Both emit PriceTick — you can fan-in to a shared handler
def on_any_tick(tick: PriceTick):
    print(f"[{tick.source}] {tick.symbol}: {tick.price:.4f}")

poly_feed.on_tick(on_any_tick)
custom_feed.on_tick(on_any_tick)

poly_feed.start()
custom_feed.start()
```

---

## Quick Reference

| What you want | Where to look |
|--------------|---------------|
| Define a new strategy | `packages/strategies/src/polymarket_algo/strategies/` |
| Define a new indicator | `packages/indicators/src/polymarket_algo/indicators/` |
| Connect a new data source | `packages/executor/src/polymarket_algo/executor/feed.py` (model after `PolymarketDataFeed`) |
| Change strategy parameters | Edit `.default_params` and `.param_grid` in the strategy class |
| Change bet sizing | `bot.py` Kelly size calc, or `Config.BET_AMOUNT` in `.env` |
| Add a copytrade filter | Extend `SelectiveFilter.should_trade()` in `packages/strategies/` |
| Run a backtest | `uv run python scripts/backtest.py` |
| Fetch fresh OHLCV data | `uv run python scripts/fetch_data.py` |
| View trade history | `uv run python scripts/history.py --stats` |
| Run tests | `uv run pytest -v` |
| Lint code | `ruff check packages/ tests/` |
| Typecheck | `ty check` |
