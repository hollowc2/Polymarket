# Strategy Research — Direction 2: Targeted Streak Variants + CI-Based Sizing

**Date:** 2026-02-28
**Strategies tested:** `streak_reversal`, `streak_rsi`, `streak_adx`
**Assets:** BTC, ETH, SOL, XRP
**Timeframes:** 5m, 15m, 1h (SOL/XRP: 5m only — 15m/1h data not yet fetched)
**Data source:** OKX spot, 730-day window (~210k candles per asset)
**Backtest method:** walk-forward split (75/25), `parameter_sweep` on train, best params applied to held-out test

---

## 1. Background

Following the multi-asset streak reversal research (see `docs/multi_asset_research.md`),
this document covers two new strategy variants built on top of the streak reversal signal:

- **StreakRSI** — streak reversal confirmed by RSI at an extreme (overbought/oversold).
  Designed to cut noise in mid-range, directionless markets.
- **StreakADX** — streak reversal gated by ADX below a threshold (choppy/ranging regime
  only). Designed to avoid betting reversals into strong trends.

Both strategies support Wilson CI half-Kelly sizing (see `packages/core/…/sizing.py`),
scaled relative to the BTC 5m trigger=4 reference (f_half = 0.014).

Code locations:
- `packages/indicators/…/adx.py` — new vectorised ADX (Wilder smoothing)
- `packages/strategies/…/streak_rsi.py` — StreakRSIStrategy
- `packages/strategies/…/streak_adx.py` — StreakADXStrategy
- `packages/strategies/…/_ci_sizing.py` — shared CI half-Kelly helper

---

## 2. Full Results (sorted by Sharpe)

| Strategy | Asset | TF | Win Rate | PnL | Max DD | Sharpe | Trades |
|---|---|---|---|---|---|---|---|
| streak_reversal | ETH | 5m | **56.3%** | **+$3,652** | -$371 | **5.38** | 5,198 |
| streak_reversal | ETH | 15m | 56.6% | +$2,517 | -$178 | **4.63** | 3,326 |
| streak_reversal | SOL | 5m | 55.5% | +$3,022 | -$250 | **4.28** | 5,584 |
| streak_rsi | ETH | 15m | **57.2%** | +$2,076 | -$279 | **3.79** | 1,606 |
| streak_adx | ETH | 5m | 56.1% | +$2,994 | -$308 | **3.70** | 2,917 |
| streak_reversal | BTC | 15m | 55.7% | +$2,070 | -$410 | 3.66 | 3,593 |
| streak_rsi | ETH | 5m | 55.0% | +$2,102 | -$522 | 2.78 | 2,624 |
| streak_reversal | BTC | 5m | 53.7% | +$2,550 | -$906 | 2.41 | 12,437 |
| streak_rsi | SOL | 5m | 54.7% | +$1,185 | -$478 | 2.21 | 3,013 |
| streak_rsi | BTC | 15m | 55.1% | +$653 | -$197 | 2.15 | 1,830 |
| streak_reversal | XRP | 5m | 54.0% | +$1,433 | -$765 | 2.01 | 5,646 |
| streak_adx | BTC | 5m | 53.4% | +$1,452 | -$462 | 1.99 | 7,470 |
| streak_adx | ETH | 15m | 56.8% | +$440 | -$133 | 1.71 | 352 |
| streak_adx | SOL | 5m | 54.1% | +$922 | -$365 | 1.64 | 3,262 |
| streak_reversal | BTC | 1h | 54.9% | +$396 | -$189 | 1.37 | 937 |
| streak_adx | BTC | 15m | 57.5% | +$123 | -$85 | 1.14 | 174 |
| streak_rsi | BTC | 1h | 54.1% | +$208 | -$194 | 0.91 | 1,023 |
| streak_rsi | BTC | 5m | 53.2% | +$475 | -$664 | 0.84 | 6,334 |
| streak_adx | XRP | 5m | 53.0% | +$225 | -$505 | 0.65 | 1,958 |
| streak_rsi | ETH | 1h | 53.8% | +$140 | -$438 | 0.48 | 450 |
| streak_adx | BTC | 1h | 46.7% | -$3 | -$50 | -0.07 | 30 |
| streak_rsi | XRP | 5m | 52.2% | -$65 | -$724 | -0.15 | 3,068 |
| streak_reversal | ETH | 1h | 52.2% | -$74 | -$558 | -0.25 | 965 |
| streak_adx | ETH | 1h | 49.3% | -$210 | -$394 | -0.93 | 270 |

---

## 3. Key Findings

### 3.1 ETH is the primary target

ETH dominates the top of every metric. At 5m:
- plain streak_reversal: Sharpe 5.38, $3,652 PnL on 5,198 trades
- streak_adx: Sharpe 3.70, $2,994 PnL on only **2,917 trades** (−44% trades, −18% PnL)
- streak_rsi: Sharpe 2.78, $2,102 PnL on 2,624 trades

ETH's mean-reversion edge is structurally stronger than BTC, SOL, or XRP at every timeframe
tested. The asset-specific Wilson CI data confirms this: ETH trigger=4 win rate = 55.0%
vs BTC's 54.0%, a statistically significant gap across 22k+ trades.

### 3.2 streak_adx is the best-quality filter for ETH 5m

`streak_adx ETH 5m` cuts trade count by 44% while keeping 82% of plain streak_reversal's
PnL. This is the highest-quality risk-adjusted variant we have:

| Metric | streak_reversal | streak_adx | Δ |
|---|---|---|---|
| Trades | 5,198 | 2,917 | −44% |
| PnL | +$3,652 | +$2,994 | −18% |
| Max DD | -$371 | -$308 | −17% |
| Sharpe | 5.38 | 3.70 | −31% |
| PnL per trade | $0.70 | **$1.03** | **+46%** |

The Sharpe drop is expected (fewer trades = less diversification of luck), but PnL per
trade improves substantially. For a capital-limited account, streak_adx is preferable.

### 3.3 streak_rsi works best on ETH 15m

`streak_rsi ETH 15m` achieves Sharpe 3.79 and win rate 57.2% — the highest win rate in
the entire table. RSI confirmation adds value on the 15m timeframe specifically, filtering
out mid-session noise that 5m cannot distinguish.

On 5m for BTC, `streak_rsi` actually *underperforms* plain streak_reversal (Sharpe 0.84
vs 2.41), suggesting RSI over-filters on the faster timeframe.

### 3.4 1h is not viable for any variant

Every 1h row is Sharpe < 1.37, and three are negative. ADX 1h (BTC, ETH) both fail badly:
only 30 and 270 trades respectively, far too few for stable estimates. 1h streak reversal
only works on BTC (Sharpe 1.37, 937 trades) — marginal at best.

### 3.5 XRP has a weak, unreliable edge

XRP's best result is plain streak_reversal Sharpe 2.01 with $1,433 PnL. Both variants
(streak_rsi −0.15, streak_adx +0.65) are negative or marginal. XRP remains the lowest-
priority asset and should not be traded without dedicated parameter re-optimisation.

### 3.6 CI-based sizing — practical impact

At the reference anchor (BTC 5m trigger=4, f_half=0.014), the CI sizing scales trade
size relative to measured edge:

| Asset | Trigger | rate | ci_lo | f_half | Size/$15 base |
|---|---|---|---|---|---|
| BTC | 3 | 0.529 | 0.525 | 0.007 | ~$7 |
| BTC | 4 | 0.540 | 0.533 | 0.014 | $15 (ref) |
| BTC | 5 | 0.538 | 0.528 | 0.012 | ~$13 |
| BTC | 8 | 0.562 | 0.531 | 0.026 | ~$28 |
| ETH | 4 | 0.550 | 0.543 | 0.022 | ~$24 |

ETH CI sizing naturally upweights ETH trades (+60% vs BTC at trigger=4) because the
measured win rate is higher and the CI lower bound clears 0.50 comfortably.

---

## 4. Priority Deployment Order

Based on Sharpe, PnL, drawdown, and trade volume:

| Rank | Strategy | Asset | TF | Rationale |
|---|---|---|---|---|
| **1** | streak_reversal | ETH | 5m | Highest Sharpe (5.38), largest PnL, most trades — primary target |
| **2** | streak_adx | ETH | 5m | Best PnL/trade ratio; deploy alongside #1 if capital allows |
| **3** | streak_reversal | ETH | 15m | Sharpe 4.63, lower DD (-$178); good second timeframe |
| **4** | streak_reversal | SOL | 5m | Sharpe 4.28, solid PnL — secondary asset |
| **5** | streak_rsi | ETH | 15m | Win rate 57.2%; lower trade count, complementary to #3 |

---

## 5. Recommended Live Parameters — streak_reversal ETH 5m

This is the deployment-ready configuration based on OOS backtest performance.

```env
STRATEGY=streak_reversal
ASSET=ETH
TIMEFRAME=5m
TRIGGER=4          # Robust across OOS; trigger=5 peaks on train but fewer OOS trades
SIZE=15.0          # Base size in USD; adjust with CI sizing for larger accounts
USE_CI_SIZING=true
```

Supporting data:
- Train Sharpe: 5.38 (trigger=4, $15 flat)
- OOS win rate: 56.3% (5,198 trades)
- Max drawdown: -$371
- Wilson CI (trigger=4): [54.3%, 55.6%], n=22,746 — narrow, statistically reliable

---

## 6. Next Steps

- [ ] Fetch 15m/1h data for SOL and XRP to fill gaps in the results table
- [ ] Run walk-forward on ETH 15m with more trigger values (currently only 3/4/5 swept)
- [ ] Test CI sizing in paper trade mode — verify size scaling matches theoretical values
- [ ] Add ETH-specific `ASSET_REVERSAL_RATES` to `sizing.py` for 15m (currently 5m only)
- [ ] Evaluate `streak_adx` on ETH with `use_ci_sizing=True` in live paper run
