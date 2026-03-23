# Design: Extract TurtleQuant + SlowQuant into Standalone Project

**Date:** 2026-03-17
**Status:** Approved

## Problem

TurtleQuant and SlowQuant are probabilistic pricing bots for longer-dated Polymarket markets. They share no runtime with the crypto up/down bots (which trade 5m/1h candle direction). Currently they live inside `crypto_up_or_down/packages/` as monorepo members, creating unnecessary coupling and making them harder to develop, run, and reason about independently.

## Goal

Extract TurtleQuant and SlowQuant into `/mnt/Files/Projects/Python/PolyMarket/turtlequant/` as a fully standalone Python project that can be installed and run without any dependency on `crypto_up_or_down`.

## Non-Goals

- Changing any logic or behavior in TurtleQuant or SlowQuant
- Publishing to PyPI
- Adding new features

---

## New Folder Structure

```
/mnt/Files/Projects/Python/PolyMarket/turtlequant/
├── pyproject.toml              # standalone project, no workspace
├── .python-version             # Python 3.13
├── .gitignore                  # covers .env, __pycache__, .venv, *.pyc, data/
├── .env.example                # API key template (see section below)
├── src/
│   └── turtlequant/
│       ├── __init__.py
│       ├── market_parser.py
│       ├── market_scanner.py
│       ├── position_manager.py
│       ├── probability_engine.py
│       ├── vol_surface.py
│       ├── data/               # duplicated from polymarket_algo.data
│       │   ├── __init__.py
│       │   └── binance.py
│       └── slowquant/
│           ├── __init__.py
│           ├── monte_carlo.py
│           ├── vol_regime.py
│           ├── opportunity_ranker.py
│           └── strategy_loop.py
├── scripts/
│   ├── turtlequant_bot.py
│   ├── slowquant_bot.py
│   ├── monitor_turtlequant.py
│   ├── monitor_slowquant.py
│   ├── calibrate_turtlequant.py
│   └── calibrate_slowquant.py
└── data/
    └── .gitignore              # gitignore runtime cache
```

---

## Import Namespace Rename

All imports in all source files and scripts — **including inline/lazy imports inside method bodies** — are updated from the monorepo namespace to the standalone namespace:

| Old Import | New Import |
|-----------|-----------|
| `from polymarket_algo.turtlequant.X import Y` | `from turtlequant.X import Y` |
| `from polymarket_algo.slowquant.X import Y` | `from turtlequant.slowquant.X import Y` |
| `from polymarket_algo.data.binance import fetch_klines` | `from turtlequant.data.binance import fetch_klines` |

Known inline imports that must not be missed:
- `vol_surface.py` line ~279: lazy `from polymarket_algo.data.binance import fetch_klines` inside a method body
- `turtlequant_bot.py` line ~358: lazy `from polymarket_algo.turtlequant.market_parser import MarketParams, OptionType`

`polymarket-algo-core` is listed as a dependency in both existing pyproject.tomls but is never imported in source — it is dropped.

### enrich_candles — do not copy

`strategy_loop.py` imports `enrich_candles` inside a `try/except ImportError` block (optional microstructure enrichment). Since `enrich.py` in turn imports `agg_trades.py`, `funding.py`, and `liquidations.py` — none of which are needed elsewhere in TurtleQuant/SlowQuant — `enrich.py` is **not** copied. The `try/except` block handles the missing import gracefully with zero-value fallbacks. No code changes required.

---

## Data Files to Duplicate

One file from `crypto_up_or_down/packages/data/src/polymarket_algo/data/` is copied into `turtlequant/src/turtlequant/data/` with its internal namespace updated:

- `binance.py` — provides `fetch_klines` (Binance OHLCV fetcher). Any `polymarket_algo.data` imports inside this file are updated to `turtlequant.data`.

A minimal `turtlequant/src/turtlequant/data/__init__.py` is created that exports `fetch_klines`.

`storage.py`, `enrich.py`, `funding.py`, `liquidations.py`, `agg_trades.py` are **not** copied.

---

## Scripts Migration

Six scripts move from `crypto_up_or_down/scripts/` to `turtlequant/scripts/`:

| Source | Destination |
|--------|-------------|
| `scripts/bots/turtlequant_bot.py` | `scripts/turtlequant_bot.py` |
| `scripts/bots/slowquant_bot.py` | `scripts/slowquant_bot.py` |
| `scripts/monitor/monitor_turtlequant.py` | `scripts/monitor_turtlequant.py` |
| `scripts/monitor/monitor_slowquant.py` | `scripts/monitor_slowquant.py` |
| `scripts/calibrate/calibrate_turtlequant.py` | `scripts/calibrate_turtlequant.py` |
| `scripts/calibrate/calibrate_slowquant.py` | `scripts/calibrate_slowquant.py` |

All imports (including inline) are updated to the new namespace.

**calibrate scripts — remove sys.path hacks:** Both `calibrate_turtlequant.py` and `calibrate_slowquant.py` contain `sys.path.insert` blocks that point at `packages/data/src`, `packages/turtlequant/src`, and `packages/slowquant/src` — monorepo-specific path manipulation to make the packages importable without installing them. These blocks must be **deleted** entirely during migration. In the standalone project, `turtlequant` is a properly installed package (`uv sync`) and no `sys.path` manipulation is needed.

---

## Cleanup in `crypto_up_or_down`

After extraction, the following are **deleted** from the monorepo:

- `packages/turtlequant/` (entire directory)
- `packages/slowquant/` (entire directory)
- `scripts/bots/turtlequant_bot.py`
- `scripts/bots/slowquant_bot.py`
- `scripts/monitor/monitor_turtlequant.py`
- `scripts/monitor/monitor_slowquant.py`
- `scripts/calibrate/calibrate_turtlequant.py`
- `scripts/calibrate/calibrate_slowquant.py`

The root `pyproject.toml` in `crypto_up_or_down` has entries removed from **three** sections:

1. `[tool.uv.workspace] members` — remove `"packages/turtlequant"` and `"packages/slowquant"`
2. `[tool.uv.sources]` — remove `polymarket-algo-turtlequant` and `polymarket-algo-slowquant` entries
3. `[tool.pytest.ini_options] pythonpath` — remove `"packages/turtlequant/src"` and `"packages/slowquant/src"`

**Nothing else in `crypto_up_or_down` changes.** `core`, `data`, `indicators`, `strategies`, `backtest`, and `executor` are untouched. `data/binance.py` and `data/enrich.py` remain in the monorepo — the up/down bots still use them.

---

## New `pyproject.toml` for Standalone Project

```toml
[project]
name = "turtlequant"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "pandas>=3.0.0",
    "numpy>=2.0.0",
    "scipy>=1.13.0",
    "requests>=2.32.5",
    "python-dateutil>=2.9.0",
    "rich>=13.0",
]

[build-system]
requires = ["hatchling>=1.25.0"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/turtlequant"]
```

---

## `.env.example`

Copy the following keys from `crypto_up_or_down/.env.example` (TurtleQuant/SlowQuant relevant keys only):

```
# Deribit IV feed (vol surface)
DERIBIT_CLIENT_ID=
DERIBIT_CLIENT_SECRET=

# Trading parameters
ENTRY_THRESHOLD=
KELLY_FRACTION=
STARTING_NAV=

# Polygon wallet (live trading only)
PRIVATE_KEY=
```

---

## Verification

After extraction, confirm:

1. `uv sync` succeeds in the new `turtlequant/` folder
2. `uv run python -c "from turtlequant import MarketScanner, VolSurface, PositionManager"` exits cleanly
3. `uv run python -c "from turtlequant.slowquant import SlowQuantRunner"` exits cleanly
4. `uv run python scripts/turtlequant_bot.py --help` exits cleanly (turtlequant_bot uses argparse)
5. `uv run pytest` in `crypto_up_or_down` still passes (no regressions)
6. `grep -r "polymarket_algo.turtlequant\|polymarket_algo.slowquant" crypto_up_or_down/packages crypto_up_or_down/scripts` returns no matches
