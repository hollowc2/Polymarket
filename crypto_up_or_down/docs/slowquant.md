# SlowQuant — Implementation Plan

## Strategy Thesis

Short-dated crypto price threshold markets on Polymarket are effectively
digital options with retail pricing. Retail traders estimate probability
by intuition; quant models compute it from volatility, time, and jump risk.
The gap is the edge.

**Best market type:** "Will BTC be above $X on Friday at 16:00 UTC?"
**Target window:** 1–10 days to expiry (6h–24h during vol spikes)
**Primary edge sources:**
1. Volatility misestimation — retail ignores σ
2. Nonlinear time decay — retail assumes linear
3. Jump risk (fat tails) — crypto has +10%/−12% moves in hours
4. Volatility acceleration lag — Polymarket reprices slowly after vol spikes

---

## Architecture

```
packages/turtlequant/     ← INFRASTRUCTURE (reused, no duplication)
  market_scanner.py       ← Gamma API, ActiveMarket, filters
  market_parser.py        ← parse_market(), MarketParams, OptionType
  vol_surface.py          ← VolSurface: Deribit IV + realized vol fallback
  position_manager.py     ← PositionManager, Position, make_position
  probability_engine.py   ← digital_probability() — used as BS pre-filter

packages/slowquant/       ← STRATEGY LAYER (all new)
  monte_carlo.py          ← Merton jump-diffusion engine + calibration
  vol_regime.py           ← Regime detector (vol accel + liq + funding)
  opportunity_ranker.py   ← Opportunity scoring + exit logic
  strategy_loop.py        ← Main event loop (SlowQuantRunner class)

scripts/
  slowquant_bot.py        ← CLI entry point
  calibrate_slowquant.py  ← Historical MC calibration: Brier + RMSE
```

### What SlowQuant imports from TurtleQuant

| Module | Usage |
|---|---|
| `market_scanner.MarketScanner` | Same Gamma API; expiry filter set by regime |
| `market_parser.parse_market()` | Same regex classifier |
| `vol_surface.VolSurface` | Same Deribit IV + realized fallback |
| `probability_engine.digital_probability()` | BS pre-filter (fast, gates MC) |
| `position_manager.PositionManager` | Different NAV limits: 1.5% / 4% / 20% |
| `position_manager.make_position()` | Direct import |

---

## Core Model: Merton Jump-Diffusion Monte Carlo

### Why not Black-Scholes?

TurtleQuant uses BS (`N(d2)`). BS assumes log-normal continuous diffusion.
Crypto has fat tails and discrete jumps (ETF news, liquidation cascades,
macro shocks). For short-dated markets, one +10% jump can flip a 30%
probability to 60%. Retail systematically underprices this.

### Model Structure

```
dS/S = (μ − λκ)dt + σdW + J dN

μ    = risk-neutral drift (r - 0.5σ² - λκ)
σ    = diffusion vol (from Deribit IV or realized vol)
dW   = Brownian increment
dN   = Poisson process (jump arrivals, intensity λ)
J    ~ Normal(μⱼ, σⱼ)   log jump size
κ    = E[eᴶ - 1]        mean jump correction
```

### Parameters

| Parameter | Typical Crypto | Notes |
|---|---|---|
| λ (jump/yr) | 15.6 (0.3/wk) | calibrated from 30d 1h returns |
| μⱼ (mean jump) | ~0 | calibrated from jump events |
| σⱼ (jump std) | 0.08 (8%) | calibrated from jump events |
| n_sims | 50,000 | <0.1s per trade |
| dt | 1 hour | `int(T_years × 8760)` steps |

### Jump Parameter Calibration

Calibrated dynamically from last 30d of 1h Binance returns:
- Jump event: `|log_return| > 4%`
- λ = jump_count / n_obs × 8760
- μⱼ, σⱼ = mean/std of jump-classified returns
- Falls back to defaults if < 5 jump events found
- Recalibrated every 12 scan cycles

### Pipeline per Market

```
BS prefilter           |BS edge| < 3%  →  skip
        ↓
Monte Carlo            50k hourly-step Merton paths
        ↓
Regime gate            mc_edge < regime.edge_threshold  →  skip
        ↓
Opportunity score      edge × liquidity × proximity × regime_multiplier
        ↓
Execute (top N)        max 5 trades per cycle
```

---

## Volatility Regime Detector

Composite signal from three sources:

```
vol_acceleration = vol_5m / vol_1h
vol_sig  = min(vol_acceleration / 4.0, 1.0)
fund_sig = min(|funding_zscore| / 3.0, 1.0)
liq_sig  = min(liq_1h_usd / (baseline × 3), 1.0)

score = 0.6×vol_sig + 0.2×fund_sig + 0.2×liq_sig
```

| Level | Score | Scan | Edge threshold | Size mult | Min expiry |
|---|---|---|---|---|---|
| normal | < 0.35 | 60s | 6% | 1.0× | 24h |
| elevated | 0.35–0.65 | 20s | 4% | 1.2× | 12h |
| spike | > 0.65 | 5s | 3% | 1.5× | 6h |

Regime is a **soft gate** (not hard). The bot always scans; regime
adjusts thresholds and scan cadence. Short-dated markets only become
eligible during elevated/spike regime.

---

## Position Sizing & Exit

### NAV Limits (tighter than TurtleQuant)

| Limit | Value | Rationale |
|---|---|---|
| Per-market | 1.5% NAV | Short-dated = higher variance |
| Per-expiry | 4% NAV | Correlated risk on same Friday |
| Total exposure | 20% NAV | Same as TurtleQuant |
| Kelly fraction | 25% | Fractional Kelly |

### Time Decay Size Scaling

| Time to expiry | Size multiplier |
|---|---|
| > 72h | 1.0× |
| 24–72h | 0.7× |
| 6–24h | 0.4× |

### Exit Triggers (priority order)

1. **Edge reversed** — `mc_prob < market_price`
2. **Edge decayed** — `current_edge < 0.4 × entry_edge`
3. **Time decay cleanup** — `< 6h remaining AND edge < 5%`

---

## Opportunity Scoring

```python
proximity   = 1.0 - |spot - strike| / spot
liq_factor  = min(liquidity_usd / 50_000, 1.0)
score       = mc_edge × liq_factor × proximity × regime.size_multiplier
```

Top 5 scores per cycle are executed. Prevents capital spray.

---

## Phase 2: Probability Curve Arbitrage

When Polymarket lists multiple strikes for the same expiry:

```
Strike    YES Price
75k       0.82
80k       0.54
85k       0.53    ← implied P(80-85k) = 1%  (absurd)
```

Curve inconsistencies → spread trade: buy 80k YES, sell 85k YES.

Module: `packages/slowquant/strike_curve.py`
- `build_market_cdf(markets)` → survival curve
- `detect_curve_violation(cdf)` → flag collapsed intervals
- `suggest_spread_trade(violation)` → (buy K1, sell K2) pair

**Not in v1.** Phase 2 only.

---

## Calibration Gate

Run `scripts/calibrate_slowquant.py` before deploying paper bot.

- Simulates 3d/7d/10d market entries on 3y of historical 1h data
- Compares MC model probability to actual outcome
- Deploy if: **Brier < 0.20** AND **RMSE < 0.05**

---

## Build Sequence

| Step | File | Depends on |
|---|---|---|
| 1 | `packages/slowquant/pyproject.toml` + `__init__.py` | — |
| 2 | `monte_carlo.py` — JumpParams, calibrate, simulate | numpy |
| 3 | `vol_regime.py` — RegimeState, get_regime | numpy |
| 4 | `opportunity_ranker.py` — Opportunity, score, exit | monte_carlo, vol_regime |
| 5 | `strategy_loop.py` — SlowQuantRunner | all above + turtlequant |
| 6 | `scripts/slowquant_bot.py` — CLI | strategy_loop |
| 7 | `scripts/calibrate_slowquant.py` — Brier/RMSE | monte_carlo, binance data |
| 8 | docker-compose service (commented out) | calibration gate pass |

---

## Deploy Sequence

```bash
# 1. Run calibration
uv run python scripts/calibrate_slowquant.py --asset btc --years 3
uv run python scripts/calibrate_slowquant.py --asset eth --years 3

# 2. Inspect: Brier < 0.20, RMSE < 0.05

# 3. Paper bot
uv run python scripts/slowquant_bot.py --paper --asset btc,eth

# 4. After 2 weeks paper trading, review history
# 5. Uncomment docker-compose service, commit+push → VPS
```

---

## Key Parameter Defaults

| Parameter | Default | CLI flag |
|---|---|---|
| Assets | btc,eth | `--asset` |
| n_sims | 50,000 | `--n-sims` |
| BS pre-filter | 3% | hardcoded |
| Min edge (normal) | 6% | `--edge-threshold` |
| Starting NAV | 100 USD | `--starting-nav` |
| Max trades/cycle | 5 | `--max-trades` |
| State dir | state/slowquant | `--state-dir` |
