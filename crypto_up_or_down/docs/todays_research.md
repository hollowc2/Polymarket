# Research Plan — 2026-03-07

## Context

We discovered the paper simulation was lying. Limit order paper trades were
filling during market resolution (ask converges to 0 on winning markets), not
from real CLOB matches. The `alt-entry(0.5)` results showing +$19 PnL and 1.42
Sharpe were fraudulent. We have no validated edge.

Actions taken today:
- Switched live bot to FOK market orders (guaranteed fill, no discount)
- Stopped live bot — no confirmed edge, not safe to trade real money
- Added `PAPER_MAX_FILL_DISCOUNT = 0.03` guard to PaperTrader to prevent future
  simulation lies
- Fixed break-even formula in `live_stats.py` (was ~54.8%, now correct ~50.1%)
- Fixed `fee_rate_bps` default from 1000 (10%) to 200 (Polymarket standard 2%)
- Retired all tainted bots; now running 4 honest paper bots only

## Currently Running (paper only)

| Bot | Strategy | Gate | State file |
|-----|----------|------|-----------|
| `streak-bot` | streak_reversal/eth/5m | none (control) | trade_history_full.json |
| `reversal-trend-bot` | streak_reversal/eth/5m | TrendFilter(ema=100) | reversal-trend-history.json |
| `adx-trend-bot` | streak_adx/eth/5m | TrendFilter(ema=100) | adx-trend-history.json |
| `rsi-session-bot` | streak_rsi/eth/15m | session:7-22 UTC | rsi-session-history.json |

Check paper results: `uv run --script scripts/live_stats.py`

---

## Research Plan

### Phase 1 — Walk-Forward Backtest (do this first)

**Why:** The existing parameter sweep selects best config on the full dataset then
reports metrics on the same data. That is not a test — it is curve fitting. The
strong Sharpe ratios (5.81, 4.42, 4.31) are almost certainly overfit.

**What to build:**
- Strict train/test split: 2022–2023 train, 2024–2025 test
- Select parameters only on the train period
- Report test-period metrics only — that is the real number
- If test Sharpe drops significantly vs train Sharpe, strategies are overfit and
  we have no edge to trade

**Files to modify:** `scripts/sweep_gates.py`, `scripts/run_all_backtests.py`

Add a `--walk-forward` flag to `sweep_gates.py` that:
1. Splits data at a configurable cutoff date (default: 2024-01-01)
2. Runs parameter sweep on train period only
3. Takes the best param combo and re-runs on test period
4. Reports both train and test Sharpe side by side

This is the highest-priority task — it could tell us the strategies have no edge
before we invest more time.

---

### Phase 2 — Validate the Signal Exists

**Why:** The streak reversal hypothesis has never been tested against actual
Polymarket resolution data. We've been assuming that because ETH streaks show
mean reversion in Binance OHLCV, the Polymarket markets are mispriced at streak
peaks. That may not be true.

**What to research:**
1. Pull historical Polymarket market resolution data from the Gamma API
2. Align with Binance OHLCV candles at market close time
3. Compute: given a 5-candle streak, what % of markets resolve against the
   streak direction?
4. If it is not meaningfully above 50%, the signal does not exist in real data

**Key question:** Does Polymarket price lag Binance? If the market is already
pricing in the streak by the time we see it, there is no edge to capture.

---

### Phase 3 — Wait for Paper Data (2–3 weeks)

The 4 paper bots are now running honest FOK simulation. Each needs ~100 settled
trades before the confidence interval is tight enough to act on.

At ~8–12 trades/day per bot: **2–3 weeks minimum**.

Do not go live before this data exists and phases 1–2 are done.

---

### Phase 4 — If Edge Confirmed, Size It Properly

Before going live again:
- Break-even is ~50.1% at fill=0.50 (fees are small — not the problem)
- Need to model realistic all-in cost: spread + FOK execution variance
- Start at 1/4 Kelly — strategies are unproven even if signal exists
- Require 100+ trades of out-of-sample paper data showing consistent positive edge

---

## Priority Order

1. **Walk-forward backtest** — Phase 1, highest priority, contained code change
2. **Polymarket resolution analysis** — Phase 2, validate signal in real data
3. **Wait for paper accumulation** — Phase 3, 2–3 weeks
4. **Live deployment only after** phases 1–3 show consistent positive signal

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/sweep_gates.py` | Gate parameter sweep — needs walk-forward mode |
| `scripts/run_all_backtests.py` | Full strategy sweep |
| `scripts/live_stats.py` | Paper bot leaderboard |
| `scripts/analyze_signal_audit.py` | Signal gap analysis (audit JSONL) |
| `packages/backtest/` | Backtest engine, walk-forward split already exists |
