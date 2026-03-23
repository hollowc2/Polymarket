# TurtleQuant Extraction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract TurtleQuant and SlowQuant from the `crypto_up_or_down` monorepo into a fully standalone project at `/mnt/Files/Projects/Python/PolyMarket/turtlequant/`.

**Architecture:** Copy source files into the new namespace (`turtlequant.*`), update all imports, move scripts, then delete the originals from the monorepo. The two projects share no code after extraction — `data/binance.py` is duplicated.

**Tech Stack:** Python 3.13, uv, hatchling, pandas, numpy, scipy, requests, rich

---

## File Map

**New files to create:**
- `turtlequant/pyproject.toml`
- `turtlequant/.python-version`
- `turtlequant/.gitignore`
- `turtlequant/.env.example`
- `turtlequant/data/.gitignore`
- `turtlequant/src/turtlequant/__init__.py`
- `turtlequant/src/turtlequant/data/__init__.py`
- `turtlequant/src/turtlequant/slowquant/` (directory)

**Files to copy + update imports:**
- `turtlequant/src/turtlequant/market_parser.py` ← `packages/turtlequant/src/polymarket_algo/turtlequant/market_parser.py`
- `turtlequant/src/turtlequant/market_scanner.py` ← `packages/turtlequant/src/polymarket_algo/turtlequant/market_scanner.py`
- `turtlequant/src/turtlequant/position_manager.py` ← `packages/turtlequant/src/polymarket_algo/turtlequant/position_manager.py`
- `turtlequant/src/turtlequant/probability_engine.py` ← `packages/turtlequant/src/polymarket_algo/turtlequant/probability_engine.py`
- `turtlequant/src/turtlequant/vol_surface.py` ← `packages/turtlequant/src/polymarket_algo/turtlequant/vol_surface.py` (1 inline import to update)
- `turtlequant/src/turtlequant/data/binance.py` ← `packages/data/src/polymarket_algo/data/binance.py` (no import changes needed)
- `turtlequant/src/turtlequant/slowquant/__init__.py` ← `packages/slowquant/src/polymarket_algo/slowquant/__init__.py`
- `turtlequant/src/turtlequant/slowquant/monte_carlo.py` ← `packages/slowquant/src/polymarket_algo/slowquant/monte_carlo.py`
- `turtlequant/src/turtlequant/slowquant/vol_regime.py` ← `packages/slowquant/src/polymarket_algo/slowquant/vol_regime.py`
- `turtlequant/src/turtlequant/slowquant/opportunity_ranker.py` ← `packages/slowquant/src/polymarket_algo/slowquant/opportunity_ranker.py`
- `turtlequant/src/turtlequant/slowquant/strategy_loop.py` ← `packages/slowquant/src/polymarket_algo/slowquant/strategy_loop.py` (1 import to update, 1 try/except left as-is)

**Scripts to copy + update:**
- `turtlequant/scripts/turtlequant_bot.py` ← `scripts/bots/turtlequant_bot.py` (6 imports to update)
- `turtlequant/scripts/slowquant_bot.py` ← `scripts/bots/slowquant_bot.py` (1 import to update)
- `turtlequant/scripts/monitor_turtlequant.py` ← `scripts/monitor/monitor_turtlequant.py` (no import changes)
- `turtlequant/scripts/monitor_slowquant.py` ← `scripts/monitor/monitor_slowquant.py` (no import changes)
- `turtlequant/scripts/calibrate_turtlequant.py` ← `scripts/calibrate/calibrate_turtlequant.py` (remove sys.path block, 2 imports to update)
- `turtlequant/scripts/calibrate_slowquant.py` ← `scripts/calibrate/calibrate_slowquant.py` (remove sys.path block, 3 imports to update)

**Files to delete from `crypto_up_or_down` after extraction:**
- `packages/turtlequant/` (entire directory)
- `packages/slowquant/` (entire directory)
- `scripts/bots/turtlequant_bot.py`
- `scripts/bots/slowquant_bot.py`
- `scripts/monitor/monitor_turtlequant.py`
- `scripts/monitor/monitor_slowquant.py`
- `scripts/calibrate/calibrate_turtlequant.py`
- `scripts/calibrate/calibrate_slowquant.py`

**Files to modify in `crypto_up_or_down`:**
- `pyproject.toml` — remove turtlequant/slowquant from 3 sections

---

## Task 1: Scaffold the Standalone Project

**Files:**
- Create: `turtlequant/pyproject.toml`
- Create: `turtlequant/.python-version`
- Create: `turtlequant/.gitignore`
- Create: `turtlequant/.env.example`
- Create: `turtlequant/data/.gitignore`

**Working directory for this task:** `/mnt/Files/Projects/Python/PolyMarket/turtlequant/`

- [ ] **Step 1: Create `pyproject.toml`**

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

- [ ] **Step 2: Create `.python-version`**

```
3.13
```

- [ ] **Step 3: Create `.gitignore`**

```
.env
.venv/
__pycache__/
*.pyc
*.pyo
data/
*.egg-info/
dist/
.ruff_cache/
```

- [ ] **Step 4: Create `.env.example`**

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

- [ ] **Step 5: Create `data/.gitignore`**

```
*
!.gitignore
```

- [ ] **Step 6: Create the source directory structure**

```bash
mkdir -p src/turtlequant/data
mkdir -p src/turtlequant/slowquant
mkdir -p scripts
```

- [ ] **Step 7: Run `uv sync` and verify it succeeds**

```bash
cd /mnt/Files/Projects/Python/PolyMarket/turtlequant
uv sync
```

Expected: resolves and installs dependencies with no errors.

- [ ] **Step 8: Commit**

```bash
git init
git add pyproject.toml .python-version .gitignore .env.example data/.gitignore uv.lock
git commit -m "feat: scaffold standalone turtlequant project"
```

---

## Task 2: Copy the `data` Subpackage

**Source:** `crypto_up_or_down/packages/data/src/polymarket_algo/data/binance.py`
**Destination:** `turtlequant/src/turtlequant/data/`

`binance.py` has **no** `polymarket_algo` internal imports — it can be copied verbatim.

- [ ] **Step 1: Write import smoke test**

Create `turtlequant/tests/test_imports.py`:

```python
def test_data_binance_import():
    from turtlequant.data.binance import fetch_klines
    assert callable(fetch_klines)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/Files/Projects/Python/PolyMarket/turtlequant
uv run pytest tests/test_imports.py::test_data_binance_import -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'turtlequant'`

- [ ] **Step 3: Copy `binance.py` verbatim**

```bash
cp /mnt/Files/Projects/Python/PolyMarket/crypto_up_or_down/packages/data/src/polymarket_algo/data/binance.py \
   src/turtlequant/data/binance.py
```

- [ ] **Step 4: Create `src/turtlequant/data/__init__.py`**

```python
from .binance import fetch_klines as fetch_klines
```

- [ ] **Step 5: Create `src/turtlequant/__init__.py`** (empty for now, will be populated in Task 3)

```python
```

- [ ] **Step 6: Install package in editable mode**

```bash
uv pip install -e .
```

- [ ] **Step 7: Run test to verify it passes**

```bash
uv run pytest tests/test_imports.py::test_data_binance_import -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/turtlequant/data/ src/turtlequant/__init__.py tests/test_imports.py
git commit -m "feat: add data subpackage with fetch_klines"
```

---

## Task 3: Copy TurtleQuant Core Modules

**Source:** `crypto_up_or_down/packages/turtlequant/src/polymarket_algo/turtlequant/`
**Destination:** `turtlequant/src/turtlequant/`

Five files: `market_parser.py`, `market_scanner.py`, `position_manager.py`, `probability_engine.py`, `vol_surface.py`.

All use **relative imports only** between themselves — no namespace change needed inside those files.

**One exception:** `vol_surface.py` has an inline import at ~line 279:
```python
from polymarket_algo.data.binance import fetch_klines
```
This must become:
```python
from turtlequant.data.binance import fetch_klines
```

- [ ] **Step 1: Add smoke tests**

Append to `tests/test_imports.py`:

```python
def test_turtlequant_core_imports():
    from turtlequant import (
        MarketScanner,
        PositionManager,
        VolSurface,
        compute_probability,
        parse_market,
    )
    assert MarketScanner is not None
    assert PositionManager is not None
    assert VolSurface is not None
    assert callable(compute_probability)
    assert callable(parse_market)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_imports.py::test_turtlequant_core_imports -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Copy all five core modules**

```bash
SRC=/mnt/Files/Projects/Python/PolyMarket/crypto_up_or_down/packages/turtlequant/src/polymarket_algo/turtlequant
DST=/mnt/Files/Projects/Python/PolyMarket/turtlequant/src/turtlequant

cp $SRC/market_parser.py     $DST/market_parser.py
cp $SRC/market_scanner.py    $DST/market_scanner.py
cp $SRC/position_manager.py  $DST/position_manager.py
cp $SRC/probability_engine.py $DST/probability_engine.py
cp $SRC/vol_surface.py        $DST/vol_surface.py
```

- [ ] **Step 4: Fix the inline import in `vol_surface.py`**

Find the line (around line 279):
```python
from polymarket_algo.data.binance import fetch_klines
```
Replace with:
```python
from turtlequant.data.binance import fetch_klines
```

- [ ] **Step 5: Update `src/turtlequant/__init__.py`**

Copy content from the original `__init__.py`, updating the module path in the docstring only — all imports are relative so they work as-is:

```python
"""TurtleQuant — Probabilistic pricing engine for Polymarket crypto option markets."""

from .market_parser import MarketParams, OptionType, parse_market
from .market_scanner import ActiveMarket, MarketScanner
from .position_manager import PositionManager
from .probability_engine import (
    barrier_down_probability,
    barrier_probability,
    compute_probability,
    digital_probability,
    european_put_probability,
)
from .vol_surface import VolSurface

__all__ = [
    "ActiveMarket",
    "MarketParams",
    "MarketScanner",
    "OptionType",
    "PositionManager",
    "VolSurface",
    "barrier_down_probability",
    "barrier_probability",
    "compute_probability",
    "digital_probability",
    "european_put_probability",
    "parse_market",
]
```

- [ ] **Step 6: Verify no remaining `polymarket_algo` references in core modules**

```bash
grep -rn "polymarket_algo" src/turtlequant/*.py
```

Expected: no output (zero matches)

- [ ] **Step 7: Run test to verify it passes**

```bash
uv run pytest tests/test_imports.py::test_turtlequant_core_imports -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/turtlequant/
git commit -m "feat: add turtlequant core modules with renamed namespace"
```

---

## Task 4: Copy SlowQuant Modules

**Source:** `crypto_up_or_down/packages/slowquant/src/polymarket_algo/slowquant/`
**Destination:** `turtlequant/src/turtlequant/slowquant/`

Four files: `monte_carlo.py`, `vol_regime.py`, `opportunity_ranker.py`, `strategy_loop.py` + `__init__.py`.

All use relative imports between themselves **except** `strategy_loop.py` which has:
- Line 25: `from polymarket_algo.data.binance import fetch_klines` → must update
- Line ~550 (inside `try/except ImportError`): `from polymarket_algo.data import enrich_candles` → **leave as-is** (graceful degradation — no `enrich.py` in the standalone project, the try/except handles it with zero-value fallbacks)

- [ ] **Step 1: Add smoke test**

Append to `tests/test_imports.py`:

```python
def test_slowquant_imports():
    from turtlequant.slowquant import (
        JumpParams,
        RegimeState,
        SlowQuantRunner,
        calibrate_jump_params,
        get_regime,
        score_opportunity,
    )
    assert SlowQuantRunner is not None
    assert callable(calibrate_jump_params)
    assert callable(get_regime)
    assert callable(score_opportunity)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_imports.py::test_slowquant_imports -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Copy all slowquant files**

```bash
SRC=/mnt/Files/Projects/Python/PolyMarket/crypto_up_or_down/packages/slowquant/src/polymarket_algo/slowquant
DST=/mnt/Files/Projects/Python/PolyMarket/turtlequant/src/turtlequant/slowquant

cp $SRC/__init__.py          $DST/__init__.py
cp $SRC/monte_carlo.py       $DST/monte_carlo.py
cp $SRC/vol_regime.py        $DST/vol_regime.py
cp $SRC/opportunity_ranker.py $DST/opportunity_ranker.py
cp $SRC/strategy_loop.py     $DST/strategy_loop.py
```

- [ ] **Step 4: Fix the top-level import in `strategy_loop.py`**

Find (line 25):
```python
from polymarket_algo.data.binance import fetch_klines
```
Replace with:
```python
from turtlequant.data.binance import fetch_klines
```

Leave the `try/except` block at line ~550 untouched — it catches `ImportError` and falls back to zeros.

- [ ] **Step 5: Verify no unexpected `polymarket_algo` references in slowquant**

```bash
grep -rn "polymarket_algo" src/turtlequant/slowquant/
```

Expected: **one match is acceptable** — the `polymarket_algo.data.enrich_candles` reference inside the `try:` block in `strategy_loop.py`. That line is intentionally left as-is for graceful degradation. Any other match is a bug that must be fixed.

- [ ] **Step 6: Run test to verify it passes**

```bash
uv run pytest tests/test_imports.py::test_slowquant_imports -v
```

Expected: PASS

- [ ] **Step 7: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all 3 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/turtlequant/slowquant/ tests/test_imports.py
git commit -m "feat: add slowquant modules with renamed namespace"
```

---

## Task 5: Copy and Fix Scripts

**Sources:** `crypto_up_or_down/scripts/bots/`, `scripts/monitor/`, `scripts/calibrate/`
**Destination:** `turtlequant/scripts/`

Import changes required per script:

| Script | Changes |
|--------|---------|
| `turtlequant_bot.py` | 6 top-level imports + 1 inline import at ~line 358 |
| `slowquant_bot.py` | 1 top-level import |
| `monitor_turtlequant.py` | None |
| `monitor_slowquant.py` | None |
| `calibrate_turtlequant.py` | Remove 2-line sys.path block + 2 imports |
| `calibrate_slowquant.py` | Remove 3-line sys.path loop + 3 imports |

- [ ] **Step 1: Copy all 6 scripts**

```bash
BOTS=/mnt/Files/Projects/Python/PolyMarket/crypto_up_or_down/scripts/bots
MON=/mnt/Files/Projects/Python/PolyMarket/crypto_up_or_down/scripts/monitor
CAL=/mnt/Files/Projects/Python/PolyMarket/crypto_up_or_down/scripts/calibrate
DST=/mnt/Files/Projects/Python/PolyMarket/turtlequant/scripts

cp $BOTS/turtlequant_bot.py   $DST/turtlequant_bot.py
cp $BOTS/slowquant_bot.py     $DST/slowquant_bot.py
cp $MON/monitor_turtlequant.py $DST/monitor_turtlequant.py
cp $MON/monitor_slowquant.py   $DST/monitor_slowquant.py
cp $CAL/calibrate_turtlequant.py $DST/calibrate_turtlequant.py
cp $CAL/calibrate_slowquant.py   $DST/calibrate_slowquant.py
```

- [ ] **Step 2: Fix `scripts/turtlequant_bot.py` — top-level imports**

Replace these 6 lines (around lines 53–58):
```python
from polymarket_algo.data.binance import fetch_klines  # noqa: E402
from polymarket_algo.turtlequant.market_parser import parse_market  # noqa: E402
from polymarket_algo.turtlequant.market_scanner import MarketScanner  # noqa: E402
from polymarket_algo.turtlequant.position_manager import PositionManager, make_position  # noqa: E402
from polymarket_algo.turtlequant.probability_engine import compute_probability  # noqa: E402
from polymarket_algo.turtlequant.vol_surface import VolSurface  # noqa: E402
```
With:
```python
from turtlequant.data.binance import fetch_klines  # noqa: E402
from turtlequant.market_parser import parse_market  # noqa: E402
from turtlequant.market_scanner import MarketScanner  # noqa: E402
from turtlequant.position_manager import PositionManager, make_position  # noqa: E402
from turtlequant.probability_engine import compute_probability  # noqa: E402
from turtlequant.vol_surface import VolSurface  # noqa: E402
```

- [ ] **Step 3: Fix `scripts/turtlequant_bot.py` — inline import at ~line 358**

Find:
```python
from polymarket_algo.turtlequant.market_parser import MarketParams, OptionType
```
Replace with:
```python
from turtlequant.market_parser import MarketParams, OptionType
```

- [ ] **Step 4: Fix `scripts/slowquant_bot.py`**

Find (line ~53):
```python
from polymarket_algo.slowquant.strategy_loop import SlowQuantRunner  # noqa: E402
```
Replace with:
```python
from turtlequant.slowquant.strategy_loop import SlowQuantRunner  # noqa: E402
```

- [ ] **Step 5: Fix `scripts/calibrate_turtlequant.py` — remove sys.path block**

Delete these two lines:
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "data" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "turtlequant" / "src"))
```

Then fix the imports:
```python
# Old:
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.turtlequant.probability_engine import digital_probability

# New:
from turtlequant.data.binance import fetch_klines
from turtlequant.probability_engine import digital_probability
```

- [ ] **Step 6: Fix `scripts/calibrate_slowquant.py` — remove sys.path block**

Delete the 3-line loop (the `_ROOT` + `for _pkg in [...]` block):
```python
_ROOT = Path(__file__).parent.parent
for _pkg in ["data", "turtlequant", "slowquant"]:
    sys.path.insert(0, str(_ROOT / "packages" / _pkg / "src"))
```

Then fix the imports:
```python
# Old:
from polymarket_algo.data.binance import fetch_klines  # noqa: E402
from polymarket_algo.slowquant.monte_carlo import calibrate_jump_params  # noqa: E402
from polymarket_algo.slowquant.monte_carlo import simulate as mc_simulate  # noqa: E402

# New:
from turtlequant.data.binance import fetch_klines
from turtlequant.slowquant.monte_carlo import calibrate_jump_params
from turtlequant.slowquant.monte_carlo import simulate as mc_simulate
```

- [ ] **Step 7: Verify zero remaining `polymarket_algo` references across all scripts**

```bash
grep -rn "polymarket_algo" scripts/
```

Expected: no output

- [ ] **Step 8: Smoke test the bot entry points**

```bash
cd /mnt/Files/Projects/Python/PolyMarket/turtlequant
uv run python scripts/turtlequant_bot.py --help
```
Expected: prints usage/help and exits 0.

```bash
uv run python -c "from turtlequant.slowquant.strategy_loop import SlowQuantRunner; print('OK')"
```
Expected: prints `OK`

```bash
uv run python scripts/calibrate_turtlequant.py --help
```
Expected: prints usage/help and exits 0.

```bash
uv run python scripts/calibrate_slowquant.py --help
```
Expected: prints usage/help and exits 0.

- [ ] **Step 9: Commit**

```bash
git add scripts/
git commit -m "feat: add migrated scripts with updated namespace"
```

---

## Task 6: Clean Up `crypto_up_or_down`

**Working directory for this task:** `/mnt/Files/Projects/Python/PolyMarket/crypto_up_or_down/`

- [ ] **Step 1: Delete turtlequant and slowquant packages**

```bash
rm -rf packages/turtlequant
rm -rf packages/slowquant
```

- [ ] **Step 2: Delete the 6 migrated scripts**

```bash
rm scripts/bots/turtlequant_bot.py
rm scripts/bots/slowquant_bot.py
rm scripts/monitor/monitor_turtlequant.py
rm scripts/monitor/monitor_slowquant.py
rm scripts/calibrate/calibrate_turtlequant.py
rm scripts/calibrate/calibrate_slowquant.py
```

- [ ] **Step 3: Update root `pyproject.toml` — remove from `[tool.uv.workspace] members`**

Remove these two lines from the `members` list:
```toml
  "packages/turtlequant",
  "packages/slowquant",
```

- [ ] **Step 4: Update root `pyproject.toml` — remove from `[tool.uv.sources]`**

Remove these two lines:
```toml
polymarket-algo-turtlequant = { workspace = true }
polymarket-algo-slowquant = { workspace = true }
```

- [ ] **Step 5: Update root `pyproject.toml` — remove from `[tool.pytest.ini_options] pythonpath`**

Remove these two lines:
```toml
  "packages/turtlequant/src",
  "packages/slowquant/src",
```

- [ ] **Step 6: Run `uv sync` to verify workspace is consistent**

```bash
uv sync
```

Expected: completes without errors referencing turtlequant or slowquant.

- [ ] **Step 7: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass. Zero failures.

- [ ] **Step 8: Verify no stale references remain**

```bash
grep -r "turtlequant\|slowquant" packages/ scripts/ tests/ pyproject.toml
```

Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add -u
git commit -m "chore: remove turtlequant and slowquant from monorepo (extracted to standalone project)"
```

---

## Final Verification

Run from `/mnt/Files/Projects/Python/PolyMarket/turtlequant/`:

```bash
uv run pytest tests/ -v
uv run python -c "from turtlequant import MarketScanner, VolSurface, PositionManager; print('TurtleQuant OK')"
uv run python -c "from turtlequant.slowquant import SlowQuantRunner; print('SlowQuant OK')"
uv run python scripts/turtlequant_bot.py --help
```

Run from `/mnt/Files/Projects/Python/PolyMarket/crypto_up_or_down/`:

```bash
uv run pytest -v
grep -r "polymarket_algo.turtlequant\|polymarket_algo.slowquant" packages/ scripts/ 2>/dev/null | grep -v __pycache__
```

Expected: all clean.
