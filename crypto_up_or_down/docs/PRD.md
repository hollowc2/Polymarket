# PRD: Polymarket Trading Toolkit

## Vision

Evolve `polymarket-streak-bot` from a single-strategy bot into a **composable, extensible Polymarket trading toolkit** — think vim/neovim or pi-mono for Polymarket trading. People can build, share, and plug in strategies, indicators, and data sources.

## What We Have (Keep Everything)

| Module | Files | Description |
|--------|-------|-------------|
| Streak Bot | `bot.py`, `strategy.py` | BTC 5-min streak reversal (mean reversion) |
| Copytrade Bot | `copybot.py`, `copybot_v2.py`, `copytrade.py`, `copytrade_ws.py` | Monitor wallets, copy BTC 5-min trades |
| Backtest | `backtest.py` | Simple streak reversal backtester |
| Infra | `polymarket.py`, `polymarket_ws.py`, `trader.py`, `blockchain.py`, `config.py`, `resilience.py` | CLOB client, WebSocket, order execution, blockchain utils |

**Nothing gets deleted.** Everything migrates into the new package structure.

## What We're Adding

### 1. Historical Data Pipeline
- **Sources:** Binance API (free, no auth for public klines)
- **Assets:** BTC, ETH
- **Timeframes:** 15m, 1h, 4h
- **Range:** 4+ years of historical candle data (OHLCV)
- **Storage:** Local CSV/Parquet files in `data/`
- **Fetcher script:** Paginate through Binance `/api/v3/klines`, save incrementally

### 2. Indicator Library
- **Built-in:** EMA, MACD, RSI, SMA, Bollinger Bands
- **Interface:** Each indicator is a function: `(candles: DataFrame) → Series/DataFrame`
- **Extensible:** Third-party indicators register via Python entry points
- **No heavy deps:** Pure numpy/pandas implementations (no TA-Lib C dependency)

### 3. Backtesting Engine
- **Input:** Historical candle data + strategy
- **Output:** Win rate, PnL curve, drawdown, Sharpe, per-trade log
- **Modes:**
  - Single run: test one strategy/params combo
  - Parameter sweep: grid search over indicator params (find optimal EMA period, RSI threshold, etc.)
  - Walk-forward: tune on window N, test on window N+1 (detect overfitting)
- **Simulates Polymarket mechanics:** Binary outcome (up/down), fixed odds (~50¢), fees
- **Fast:** Vectorized pandas operations, not candle-by-candle Python loops

### 4. Candle Direction Strategy (The New Edge)
- Predict next candle direction (up/down) using indicator confluence
- **Indicators:** EMA crossover + MACD signal + RSI zones + price action
- **Timeframes:** 1h and 4h (dodge latency issues)
- **Position sizing:** Default 15 shares, 20 when indicators strongly align
- **Target:** 55%+ win rate out-of-sample (validated via walk-forward backtest)

### 5. Plugin System
- **Strategy plugins:** Implement a `Strategy` protocol, register via entry points
- **Indicator plugins:** Same pattern
- **Data source plugins:** Add new exchanges/data sources
- **Distribution:** `uv pip install polymarket-strategy-xyz` → auto-discovered

## New Repo Structure

```
polymarket-streak-bot/          # keep repo name (stars + links)
├── pyproject.toml              # uv workspace root
├── README.md                   # updated: "Polymarket Trading Toolkit"
│
├── packages/
│   ├── core/                   # shared types, config, plugin registry
│   │   ├── pyproject.toml
│   │   └── src/polymarket_algo/core/
│   │       ├── types.py        # Candle, Signal, Trade, Strategy protocol
│   │       ├── config.py       # unified config (from existing config.py)
│   │       └── plugin.py       # entry point discovery
│   │
│   ├── data/                   # market data fetching & storage
│   │   ├── pyproject.toml
│   │   └── src/polymarket_algo/data/
│   │       ├── binance.py      # Binance kline fetcher
│   │       ├── polymarket.py   # existing Polymarket data (migrated)
│   │       └── storage.py      # CSV/Parquet read/write
│   │
│   ├── indicators/             # technical indicators
│   │   ├── pyproject.toml
│   │   └── src/polymarket_algo/indicators/
│   │       ├── ema.py
│   │       ├── macd.py
│   │       ├── rsi.py
│   │       └── ...
│   │
│   ├── strategies/             # built-in strategies
│   │   ├── pyproject.toml
│   │   └── src/polymarket_algo/strategies/
│   │       ├── streak_reversal.py    # migrated from strategy.py
│   │       ├── copytrade.py          # migrated from copytrade.py
│   │       ├── candle_direction.py   # NEW: indicator-based prediction
│   │       └── ...
│   │
│   ├── backtest/               # backtesting engine
│   │   ├── pyproject.toml
│   │   └── src/polymarket_algo/backtest/
│   │       ├── engine.py       # core backtest loop
│   │       ├── metrics.py      # winrate, sharpe, drawdown, etc.
│   │       ├── sweep.py        # parameter grid search
│   │       └── report.py       # output formatting
│   │
│   └── executor/               # live trading execution
│       ├── pyproject.toml
│       └── src/polymarket_algo/executor/
│           ├── polymarket.py   # migrated CLOB client
│           ├── ws.py           # migrated WebSocket
│           ├── trader.py       # migrated order logic
│           └── blockchain.py   # migrated blockchain utils
│
├── data/                       # downloaded historical data (gitignored)
│   ├── btc_1h.parquet
│   ├── eth_1h.parquet
│   └── ...
│
├── scripts/
│   ├── fetch_data.py           # download historical candles
│   ├── backtest.py             # CLI: run backtests
│   ├── bot.py                  # CLI: run live bot
│   └── copybot.py              # CLI: run copytrade bot
│
└── examples/
    └── custom_strategy/        # example plugin for community
        ├── pyproject.toml
        └── src/my_strategy/
            └── strategy.py
```

## Strategy Protocol

```python
from typing import Protocol
import pandas as pd

class Strategy(Protocol):
    """Interface all strategies must implement."""
    
    name: str
    timeframe: str  # "5m", "15m", "1h", "4h"
    
    def setup(self, config: dict) -> None:
        """Initialize with config params."""
        ...
    
    def evaluate(self, candles: pd.DataFrame) -> Signal:
        """Given historical candles, return a trading signal."""
        ...
    
    def size(self, signal: Signal, bankroll: float) -> float:
        """Position sizing given a signal and current bankroll."""
        ...
```

## CLI Interface

```bash
# Fetch data
polymarket-algo fetch --asset btc eth --timeframe 1h 4h --years 4

# Backtest
polymarket-algo backtest --strategy candle-direction --asset eth --timeframe 1h
polymarket-algo backtest --strategy streak-reversal --data polymarket_resolved.json
polymarket-algo backtest --strategy candle-direction --sweep --param rsi_period=10:20 --param ema_fast=5:15

# Live trading
polymarket-algo run --strategy candle-direction --timeframe 1h --paper
polymarket-algo run --strategy streak-reversal --paper
polymarket-algo run --strategy copytrade --wallets 0x123,0x456

# List available strategies/indicators
polymarket-algo list strategies
polymarket-algo list indicators
```

## Tech Stack

- **Python 3.13+**
- **uv** — package management, workspaces, tool distribution
- **pandas + numpy** — data processing & vectorized backtesting
- **polars** (optional) — faster alternative for large datasets
- **py-clob-client** — Polymarket CLOB API (existing)
- **web3** — blockchain interactions (existing)
- **websockets** — real-time data (existing)
- **click or typer** — CLI framework

## Phases

### Phase 1: Foundation (Now)
- [ ] Fetch 4yr BTC+ETH historical data (15m, 1h, 4h) from Binance
- [ ] Build backtesting engine with metrics (winrate, PnL, drawdown, Sharpe)
- [ ] Implement indicator library (EMA, MACD, RSI)
- [ ] Build candle direction strategy
- [ ] Validate: achieve 55%+ OOS winrate on 1h ETH candles

### Phase 2: Refactor (After validation)
- [ ] Restructure repo into `packages/` monorepo
- [ ] Migrate existing streak bot + copytrade into strategy plugins
- [ ] Define Strategy protocol + plugin entry points
- [ ] Set up uv workspaces
- [ ] Update README + docs

### Phase 3: Polish & Ship
- [ ] CLI interface (`polymarket-algo` command)
- [ ] Parameter sweep / walk-forward validation
- [ ] Example plugin for community
- [ ] GitHub Actions CI
- [ ] Live paper trading integration for new strategies

## Non-Goals (For Now)
- ML/AI-based prediction (keep it simple: indicators + math)
- Multi-exchange support (Polymarket only)
- Web UI / dashboard
- Hosted service
