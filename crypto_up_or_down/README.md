# Polymarket Trading Toolkit

Composable Python toolkit for backtesting and live execution on Polymarket BTC/ETH/SOL/XRP prediction markets.

> **Disclaimer:** Experimental software. Paper trade first. Never default to live.

## Architecture

```text
packages/
  core/           → Protocol types (Strategy, Indicator, DataFeed, PriceTick), config, plugin registry
  data/           → Binance OHLCV fetcher + Parquet storage; CVD, liquidations, funding enrichment
  indicators/     → EMA, SMA, RSI, MACD, Bollinger Bands (pure numpy/pandas)
  strategies/     → Streak reversal, APEX Hybrid, ApexML, microstructure strategies + gates
  backtest/       → Engine, parameter sweep, walk-forward validation, metrics
  executor/       → Polymarket CLOB client, WebSocket feeds, PaperTrader, LiveTrader, resilience
  turtlequant/    → Digital option pricing for longer-term Polymarket markets (European/barrier)
  slowquant/      → Short-dated vol-regime strategy layer on top of TurtleQuant (Merton jump-diffusion)

scripts/
  bots/           → Long-running bot runners (Docker-deployed)
  monitor/        → Live dashboards and monitoring tools
  backtest/       → Backtest, sweep, and training scripts
  calibrate/      → Calibration and parameter sweep scripts
  analysis/       → One-shot analysis and comparison tools
  utils/          → Data fetch, wallet, and maintenance utilities
docs/             → Architecture, conventions, decisions
```

## Setup

**Requires:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo>
cd crypto_up_or_down
uv sync --all-packages
cp .env.example .env

# Optional: install git hooks
prek install
```

**With Nix:**
```bash
nix develop    # auto-runs uv sync + prek install
```

## Usage

### Fetch Data

```bash
uv run python scripts/utils/fetch_data.py
```

### Backtesting

```bash
# Parameter sweep + walk-forward
uv run python scripts/backtest/backtest.py

# Gate sweep (TrendFilter, VolatilityGate, etc.)
uv run python scripts/calibrate/sweep_gates.py --walk-forward --cutoff 2024-01-01

# APEX Hybrid sweep
uv run python scripts/backtest/backtest_apex.py --hybrid
```

### Live / Paper Trading

```bash
# Streak reversal bot (paper)
uv run python scripts/bots/streak_bot.py --paper

# APEX Hybrid bot (paper)
uv run python scripts/bots/apex_bot.py --paper

# TurtleQuant option pricing bot (paper)
uv run python scripts/bots/turtlequant_bot.py --paper

# SlowQuant vol-regime bot (paper)
uv run python scripts/bots/slowquant_bot.py --paper
```

## Strategies

| Strategy | Description | Best Result (walk-forward) |
|---|---|---|
| `StreakReversal` | Bet against consecutive same-direction candles | Sharpe 6.60 (ETH/5m + TrendFilter) |
| `StreakADX` | Streak reversal filtered by ADX trend strength | — |
| `StreakRSI` | Streak reversal filtered by RSI | — |
| `ApexHybridStrategy` | Streak reversal + TFI exhaustion confirmation | Sharpe 6.54 (ETH/5m) |
| `ApexMLStrategy` | Walk-forward logistic regression on 12 microstructure features | train pre-2024 / test 2024+ |
| `CVDDivergence` | Price/CVD divergence signals | — |
| `LiquidationCascade` | Cascading liquidation entry | — |
| `FundingRateExtremes` | Extreme funding rate reversals | — |

**Gates:** `TrendFilter`, `VolatilityGate`, `VolumeFilter`, `SessionFilter`

Best confirmed gate: `TrendFilter(ema_period=50, mode="veto_with_trend")` on ETH/5m.

## Strategy Protocol

All strategies must implement:

```python
class MyStrategy:
    name: str
    description: str
    timeframe: str

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        # Returns DataFrame with "signal" (1=UP, -1=DOWN, 0=skip) and "size" (float)
        ...

    @property
    def default_params(self) -> dict: ...

    @property
    def param_grid(self) -> dict[str, list]: ...
```

See `examples/custom_strategy/` for a minimal plugin example.

## Plugin System

Strategies and indicators are auto-discovered via:
- **Entry points:** `polymarket_algo.strategies` / `polymarket_algo.indicators`
- **Local drop-ins:** `~/.polymarket-algo/plugins/*.py`

## Development

```bash
ruff check packages/ tests/     # Lint
ruff format --check packages/   # Format check
ty check                        # Typecheck
uv run pytest -v                # Tests
```

Git hooks (pre-commit: ruff check + format; pre-push: ty typecheck) installed via `prek install`.

## License

MIT
