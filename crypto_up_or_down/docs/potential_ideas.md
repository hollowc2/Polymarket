# Potential Strategy Ideas — Polymarket Auto Trader

**Date**: 2026-03-08
**Author**: Research synthesis

---

## Executive Summary

Current production best: `StreakReversal + TrendFilter(ema=100)` on ETH/5m, Sharpe **7.96** (walk-forward test 2024–2026). Secondary: `ApexHybrid` on ETH/5m (Sharpe 6.54) and `ApexML` (logistic walk-forward).

Key confirmed insights:
- **TFI/OBI are REVERSAL signals at 5m** (corr_fwd = −0.025) — not momentum
- **15m momentum is the strongest single feature** (corr_fwd = +0.2179 for mom_15m)
- CVD, liquidations, and funding are all accessible via `enrich_candles()`
- Polymarket markets resolve every 5m/15m/1h — strategy must have edge within that window

The 15 ideas below are novel additions to the existing strategy suite, ordered by implementation confidence.

---

## Tier 1: High Confidence — Novel + Buildable + Strong Edge Theory

These are ready to implement. Each has clear data requirements met by the existing pipeline, a testable hypothesis, and strong academic grounding.

---

### 1. VPIN Toxicity Gate (`VPINGate`)

**Core idea**: Volume-Synchronized Probability of Informed Trading (VPIN) measures order flow toxicity. High VPIN means informed traders are dominating order flow — adverse selection risk is elevated and fading is dangerous. Low VPIN means uninformed/noise flow — safer to enter contrarian positions.

**Signal construction**:
```
For each candle bucket i:
    bucket_imbalance_i = |buy_vol_i - sell_vol_i| / total_vol_i

VPIN_t = rolling_mean(bucket_imbalance, window=30)
```

**Gate logic**:
- VPIN > 0.6 → **VETO** all signals (toxic, informed flow dominates)
- VPIN 0.3–0.6 → allow signals at reduced size (×0.7)
- VPIN < 0.3 → allow all signals at full size (uninformed flow, safe to fade)

**Academic basis**: Easley, López de Prado, O'Hara (2012) — VPIN predicts Flash Crash 65 minutes ahead. Crypto-specific: VPIN predicts price jumps and is a leading indicator of adverse selection on limit-order-book venues. Particularly relevant for Polymarket where CLOB imbalance is already a confirmed signal.

**Data requirements**: `buy_vol`, `sell_vol` columns — already in `enrich_candles()` output.

**Implementation**:
```python
class VPINGate:
    name = "vpin_gate"
    def __init__(self, window: int = 30, toxicity_threshold: float = 0.6, safe_threshold: float = 0.3):
        ...
    def filter(self, candles: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        imbalance = (candles["buy_vol"] - candles["sell_vol"]).abs() / candles["volume"]
        vpin = imbalance.rolling(self.window).mean()
        signals.loc[vpin > self.toxicity_threshold, "signal"] = 0
        signals.loc[(vpin > self.safe_threshold) & (vpin <= self.toxicity_threshold), "size"] *= 0.7
        return signals
```

**Register in**: `packages/strategies/src/polymarket_algo/strategies/__init__.py`
**Test on**: 5m and 15m, all 4 assets
**Expected improvement**: +0.3–0.5 Sharpe when stacked on StreakReversal (gates out toxic periods)

---

### 2. HMM Regime Router (`HMMRegimeRouter`)

**Core idea**: A 2-state Hidden Markov Model on realized volatility classifies market regime as CALM or TURBULENT. Different parameter sets are optimal per regime — CALM favors more sensitive streak detection; TURBULENT favors conservative sizing or abstention.

**Signal construction**:
```
vol_t = rolling_std(returns, window=20)
states = Viterbi(HMM(n_states=2), vol_t)
# State 0 = CALM (low vol), State 1 = TURBULENT (high vol)
```

**Strategy logic**:
- CALM regime → `StreakReversal(trigger=3, size=20)` (aggressive)
- TURBULENT regime → `StreakReversal(trigger=5, size=10)` (conservative)
- Extreme TURBULENT (top 10% vol) → no trade

**Wrapper pattern**:
```python
class HMMRegimeRouter:
    name = "hmm_regime_router"
    def __init__(self, calm_strategy, turbulent_strategy, vol_window=20):
        ...
    def evaluate(self, candles, **params):
        vol = candles["close"].pct_change().rolling(self.vol_window).std()
        states = self._fit_hmm(vol)
        result = pd.DataFrame(index=candles.index)
        calm_mask = states == CALM_STATE
        result.loc[calm_mask] = self.calm_strategy.evaluate(candles[calm_mask], **params)
        result.loc[~calm_mask] = self.turbulent_strategy.evaluate(candles[~calm_mask], **params)
        return result
```

**Academic basis**: Springer Nature (2024) — regime-switching HMM strongly preferred over global models for financial time series; HAR+HMM outperforms single-regime models. Crypto vol regimes are well-separated (BitMEX data shows bimodal vol distribution).

**Dependencies**: `hmmlearn` (pure scipy/numpy, lightweight) OR manual Baum-Welch (50 lines).
**Add to**: `packages/strategies/pyproject.toml` — `hmmlearn>=0.3`
**Test on**: 15m, 1h (need enough history for vol estimation); also 5m with 50-period lookback.
**Assets**: All 4

---

### 3. BTC Lead-Lag Cross-Asset Signal (`BTCLeadLagStrategy`)

**Core idea**: BTC leads ETH/SOL/XRP by 1–2 candles at 5m timeframe. BTC price discovery precedes altcoin repricing. Using BTC's N-period momentum as a leading indicator for altcoin direction captures this structural lag.

**Signal construction**:
```
btc_mom_3bar = (btc_close_t - btc_close_{t-3}) / btc_close_{t-3}
btc_mom_zscore = zscore(btc_mom_3bar, window=50)

alt_return_1bar = (alt_close_t - alt_close_{t-1}) / alt_close_{t-1}
btc_return_1bar = (btc_close_t - btc_close_{t-1}) / btc_close_{t-1}

lagging = alt_return_1bar < btc_return_1bar * 0.5
```

**Entry logic**:
- `btc_mom_zscore > +1.5` AND target altcoin is lagging (hasn't fully repriced) → signal = +1 (bet UP on alt)
- `btc_mom_zscore < -1.5` AND target altcoin is ahead of BTC drop → signal = −1 (bet DOWN on alt)
- No trade if altcoin has already fully repriced (lead-lag window closed)

**Novel angle**: Apply primarily to SOL and XRP where correlation to BTC is higher and liquidity slightly lower (slower price discovery). ETH is best-correlated but also most liquid, so lag window is shorter.

**Academic basis**: Asymmetric lead-lag confirmed: BTC→ETH information transfer faster than reverse (Repec 2019). CME 2023 study: BTC leads altcoin repricing by 1–3 5m candles consistently.

**Data requirements**: Requires fetching BTC OHLCV alongside altcoin OHLCV — structurally feasible with existing `fetch_klines()`.

**Implementation note**: `evaluate(candles, btc_candles=None, **params)` — pass BTC candles as extra kwarg. Backtest requires aligned BTC + target candle DataFrames.

**Test on**: 5m, 15m. Assets: ETH, SOL, XRP (BTC as leading indicator).

---

### 4. Volatility-Adjusted Streak (`AdaptiveStreakStrategy`)

**Core idea**: Current streak reversal uses fixed trigger (3–4 candles). The appropriate trigger depends on volatility regime: in low-vol markets, 3 same-direction closes is already meaningful exhaustion; in high-vol markets, 6 in a row may not be exhausted yet.

**Signal construction**:
```
atr_14 = average_true_range(candles, period=14)
atr_pctile = rolling_rank(atr_14, window=200)  # percentile 0..1

trigger_map:
    atr_pctile < 0.25 → trigger = 3 (low vol, sensitive)
    atr_pctile 0.25–0.75 → trigger = 4 (normal)
    atr_pctile 0.75–0.90 → trigger = 5 (high vol, conservative)
    atr_pctile > 0.90 → trigger = None (no trade, extreme vol)
```

**Position sizing extension (Kelly-inspired)**:
```
target_vol = 0.02  # 2% per-position target vol
position_size = base_size × (target_vol / current_atr_normalized)
position_size = clip(position_size, min_size, max_size)
```

**Academic basis**: GARCH literature establishes vol clustering — quiet periods follow quiet, volatile follows volatile. Adjusting signal sensitivity to realized vol is core adaptive strategy design. ATR percentile is a non-parametric vol regime indicator that doesn't require parameter fitting.

**Implementation**: Extend `StreakReversalStrategy` with `adaptive_trigger: bool = False` parameter. When True, override trigger from param_grid with ATR-derived value.

**Test on**: 5m, 15m. Assets: All 4.
**Expected improvement**: Primarily reduces drawdown in high-vol periods without sacrificing edge in low-vol.

---

### 5. Multi-Asset Streak Confluence (`MultiAssetStreakStrategy`)

**Core idea**: When BTC, ETH, SOL, and XRP simultaneously show long down-streaks, the reversal signal on any single asset is stronger than when that asset streaks in isolation. Cross-asset confluence filters out idiosyncratic noise.

**Signal construction**:
```
For each asset a in {BTC, ETH, SOL, XRP}:
    streak_a = consecutive_same_direction_close_count(candles_a)

confluence_score = count(|streak_a| >= trigger for a in assets) / 4

Entry on target asset when:
    target_streak >= trigger  AND  confluence_score >= 0.5 (at least 2 of 4 agree)

Size multiplier: size × (1 + confluence_score)
```

**Novel angle**: Exploits cross-asset synchrony during correlated sell-offs and relief rallies. A BTC down-streak with ETH/SOL/XRP also streaking down signals systemic move → reversal when overextended is high-probability.

**Data requirements**: Multi-asset OHLCV already available via `fetch_klines()`. Need to pass all 4 asset candle DataFrames into `evaluate()`.

**Implementation note**: `evaluate(candles, all_asset_candles: dict[str, pd.DataFrame], **params)` — multi-input evaluate signature.

**Test on**: 5m (synchrony strongest at 5m). Assets: Any 1 target, using others as confluence.

---

### 6. Cross-TF Trend Hierarchy Signal (`TFHierarchyStrategy`)

**Core idea**: Trade in the direction of the dominant trend on higher timeframes. A 5m streak reversal UP is stronger if 15m and 1h trends are also UP. Use a 3-TF cascade: 1h trend sets bias → 15m momentum confirms → 5m streak provides entry.

**Signal construction**:
```
1h level: EMA(50) direction → price > ema_50_1h means UPTREND
15m level: 5-period return direction → sign(close_t - close_{t-5})
5m level: streak reversal signal (standard StreakReversal logic)

Entry conditions (strict):
    signal = +1 (bet UP) only if:
        - 5m streak DOWN reversed to first UP candle  (reversal entry)
        - 15m 5-period momentum is UP
        - 1h EMA(50) is pointing UP (price > EMA)

    signal = -1 (bet DOWN) only if:
        - 5m streak UP reversed to first DOWN candle
        - 15m 5-period momentum is DOWN
        - 1h EMA(50) is pointing DOWN (price < EMA)

    Otherwise: signal = 0 (no trade — fighting higher TF)
```

**Academic basis**: TF hierarchy is standard in institutional FX/futures ("multiple timeframe analysis"). ApexML research confirmed mom_15m (corr +0.2179) as strongest single feature — this strategy operationalizes that finding as an explicit gate. Multi-TF confluence reduces false entry rate significantly.

**Implementation**: Requires fetching 15m and 1h candles alongside 5m. `evaluate(candles_5m, candles_15m, candles_1h, **params)`.

**Test on**: 5m entry with 15m + 1h filter. Assets: All 4.
**Expected improvement**: Higher win rate than raw streak reversal at cost of reduced trade frequency (quality > quantity).

---

## Tier 2: Strong Potential — More Complex to Build

These have strong theoretical backing but require either more complex data processing, new dependencies, or careful feature engineering.

---

### 7. GARCH Volatility Forecast Gate (`GARCHVolFilter`)

**Core idea**: Fit GARCH(1,1) to rolling returns. Forecast next-period conditional variance. If forecast_vol > 95th percentile of historical vol → avoid trading. Scale position size inversely with conditional volatility.

**Signal construction**:
```
GARCH(1,1): sigma_t+1^2 = omega + alpha*epsilon_t^2 + beta*sigma_t^2

where:
    epsilon_t = return_t - mu
    sigma_t = conditional std dev

Position sizing:
    target_vol = 0.015  # 1.5% per-trade target vol
    size = base_size × (target_vol / sigma_{t+1})
    size = clip(size, 0.05, 0.25)  # 5% to 25% of bankroll

Gate:
    if sigma_{t+1} > percentile_95(historical_sigma): size = 0
```

**Academic basis**: GARCH volatility clustering well-documented in crypto (Katsiampa 2017). EGARCH captures leverage effect (negative shocks have larger impact than positive). EWMA is competitive for risk estimation and simpler to implement.

**Implementation**: Lightweight GARCH(1,1) in pure numpy (no `arch` or `statsmodels` needed). Update parameters daily via MLE (not candle-by-candle — too slow). Cache fitted parameters in memory.

**Complexity note**: Parameter estimation (MLE) needs scipy.optimize or gradient descent. Manual implementation is ~100 lines. Alternatively, use EWMA as approximation: `sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * epsilon_t^2` with `lambda=0.94` (RiskMetrics standard).

**Dependencies**: scipy (already likely present); OR pure numpy EWMA.
**Test on**: 5m, 15m. Assets: All 4.

---

### 8. RSI Divergence Detector (`RSIDivergenceStrategy`)

**Core idea**: Classic RSI divergence — price makes new swing high but RSI makes lower high = bearish divergence (momentum fading). Reliable reversal signal, particularly at RSI extremes.

**Signal construction**:
```
lookback = 10  # candles to look back for swing pivot

Bearish divergence:
    price_high_now > price_high_{t-lookback..t}  (price making new high)
    AND rsi_now < rsi_{t-lookback..t at price_high}  (RSI lower)
    AND rsi_now > 65  (in overbought zone)
    → signal = -1

Bullish divergence:
    price_low_now < price_low_{t-lookback..t}  (price making new low)
    AND rsi_now > rsi_{t-lookback..t at price_low}  (RSI higher)
    AND rsi_now < 35  (in oversold zone)
    → signal = +1
```

**Swing detection**: Use `argrelextrema(candles["close"], np.greater, order=3)` for local maxima/minima detection.

**Academic basis**: RSI divergence validated in multiple crypto intraday backtests. Filtering by RSI extreme zone (not just divergence alone) significantly reduces false positives. More reliable on 15m+ than 5m (less noise).

**Dependencies**: `numpy` argrelextrema (already available). RSI already in `packages/indicators/`.
**Test on**: 15m, 1h (5m too noisy for meaningful divergence pivots). Assets: ETH, BTC.

---

### 9. VWAP Mean Reversion (`VWAPReversionStrategy`)

**Core idea**: VWAP acts as a daily price anchor. Extreme deviations from intraday VWAP tend to revert. On Polymarket 5m/15m markets, large intraday dislocations from VWAP often revert within 1–3 candles.

**Signal construction**:
```
# Reset VWAP at UTC 00:00 each day
vwap_t = cumsum(close × volume) / cumsum(volume)  [reset daily]

# VWAP deviation z-score (rolling 96-period ≈ 8h of 5m candles)
vwap_dev = close - vwap_t
dev_zscore = (vwap_dev - rolling_mean(vwap_dev, 96)) / rolling_std(vwap_dev, 96)

Signal:
    dev_zscore > +2.0 → signal = -1 (price too far above VWAP, bet DOWN)
    dev_zscore < -2.0 → signal = +1 (price too far below VWAP, bet UP)

Confirmation filter (reduce false entries):
    Only enter if current candle is first reversal candle (close moves toward VWAP)
    i.e., for bet DOWN: close_t < close_{t-1}
```

**Academic basis**: VWAP reversion is a benchmark in institutional execution algorithms. "VWAP deviation shows asymmetric mean reversion" (microstructure literature). Particularly strong in range-bound Asian session (00:00–08:00 UTC).

**Data requirements**: OHLCV only. No enrichment needed. Volume required.

**Enrichment addition**: Add `vwap` and `vwap_dev_zscore` columns to `enrich_candles()` output.

**Test on**: 5m, 15m. Assets: All 4. **Best combined with SessionFilter** (strongest in Asian session).

---

### 10. Intraday Seasonal Betting (`IntraSeasonalStrategy`)

**Core idea**: Crypto has documented intraday seasonality — US session (13:00–21:00 UTC) is directional; Asian session (00:00–08:00 UTC) is range-bound; London open (07:00–10:00 UTC) has a volatility spike. Use time-of-day return distributions to bet against extreme hourly moves.

**Signal construction**:
```
# Build historical distribution per hour-of-day
For each hour h in 0..23:
    hist_returns_h = all historical 15m returns where hour == h
    mu_h = mean(hist_returns_h)
    sigma_h = std(hist_returns_h)

# Current bar z-score relative to this hour's distribution
current_return = (close_t - close_{t-1}) / close_{t-1}
zscore_h = (current_return - mu_h) / sigma_h

Signal:
    zscore_h > +2.0 → signal = -1 (extreme up move for this hour → revert)
    zscore_h < -2.0 → signal = +1 (extreme down move for this hour → revert)
    |zscore_h| < 1.5 → signal = 0 (normal range for this hour)

Filter:
    Only trade hours with significant historical sample (>500 bars)
    Skip hours 04:00–07:00 UTC (dead zone, low liquidity)
```

**Academic basis**: Intraday seasonality in crypto documented (Admati & Pfleiderer 1988 framework applied to 24/7 markets). Liquidity cycles follow institutional participation hours. Mean reversion of extreme moves against the hour's typical distribution is a clean statistical edge.

**Enrichment needed**: Add `hour_of_day` and `day_of_week` columns to enriched candle data (trivial datetime extraction).
**Test on**: 15m, 1h. Assets: BTC, ETH (more liquid, cleaner seasonal patterns).
**Training requirement**: Needs 90+ days of data to build reliable per-hour distributions.

---

### 11. OI-CVD Divergence (`OICVDDivergenceStrategy`)

**Core idea**: When Open Interest rises (crowded positioning) but CVD is negative (selling flow), longs are being added against selling pressure — fragile positioning that tends to unwind sharply. Vice versa for short squeezes.

**Signal construction**:
```
# Use funding_zscore as OI/positioning proxy
# funding_zscore > 0 = crowded longs, < 0 = crowded shorts

cvd_5bar = sum(delta, window=5)  # 5-bar cumulative CVD

Bearish divergence (long squeeze setup):
    funding_zscore > +1.5  (crowded longs)
    AND cvd_5bar < -threshold  (selling flow despite bullish positioning)
    AND streak_down >= 2  (price already showing weakness)
    → signal = -1 (bet DOWN, liquidation cascade likely)

Bullish divergence (short squeeze setup):
    funding_zscore < -1.5  (crowded shorts)
    AND cvd_5bar > +threshold  (buying flow despite bearish positioning)
    AND streak_up >= 2
    → signal = +1 (bet UP, short squeeze likely)

threshold = rolling_std(delta, 50) × 1.5  # dynamic threshold
```

**Academic basis**: OI + CVD divergence validates leverage unwind mechanics in crypto. High OI with one-sided funding + opposing CVD = liquidation cascade risk elevated. Related to `LiquidationCascade` strategy but uses positioning proxy rather than realized liquidation events.

**Data requirements**: `funding_zscore`, `delta`, `cvd` — all in `enrich_candles()`.
**Test on**: 15m, 1h (funding updates every 8h; CVD needs time to show divergence). Assets: BTC, ETH.

---

### 12. CatBoost Multi-Feature ML (`CatBoostMLStrategy`)

**Core idea**: Upgrade from ApexML's logistic regression to CatBoost (gradient boosting). Capture nonlinear feature interactions that logistic regression cannot. 20+ features including all ApexML features plus VWAP deviation, ATR percentile, seasonal features, and technical indicators.

**Feature set (20 features)**:
```
Microstructure (from ApexML):
    tfi, obi, microprice_drift, hawkes_intensity, cascade_score, funding_zscore

Multi-TF momentum (from ApexML):
    mom_15m, mom_60m, cvd_15m, cvd_60m, trade_15m, trade_60m

NEW additions:
    vwap_dev_zscore   — VWAP deviation z-score
    atr_percentile    — ATR percentile vs 200-period window
    hour_sin          — sin(2π × hour / 24)  (cyclical encoding)
    hour_cos          — cos(2π × hour / 24)
    day_of_week       — 0..6 (categorical, CatBoost handles natively)
    vpin_estimate     — rolling |buy_vol - sell_vol| / volume
    streak_length     — current streak count (signed: +3 = 3 up, −3 = 3 down)
    rsi_14            — 14-period RSI
    bb_width          — Bollinger Band width (upper-lower)/middle
    volume_zscore     — rolling z-score of volume
```

**Training protocol**:
```
Same walk-forward as ApexML:
    Train: pre-2024 data
    Test: 2024+ data
    CatBoost params: iterations=1000, learning_rate=0.05, depth=6
    loss_function="Logloss", eval_metric="AUC"
    early_stopping_rounds=200
    use_best_model=True
```

**Inference**: Export model to JSON (CatBoost native export) OR extract leaf weights to numpy arrays for deployment (no catboost dependency at runtime).

**Academic basis**: arXiv 2602.00776 — XGBoost/CatBoost on microstructure features; gradient boosting outperforms logistic regression on nonlinear feature interactions. CatBoost stable on categorical features (day_of_week) without manual encoding.

**Dependencies**: `catboost` for training only. Inference: numpy from exported weights.
**Add to training env**: `packages/strategies/pyproject.toml` — `catboost>=1.2` (optional/dev dependency)
**Test on**: 5m, 15m. Assets: All 4.

---

## Tier 3: Exploratory / Longer Development

These require either raw tick data, novel data sources, or significant implementation effort.

---

### 13. Neural Hawkes Intensity (`HawkesMMStrategy`)

**Core idea**: Fit a parametric Hawkes process to raw trade timestamps (not candle-aggregated). Extract the "endogeneity ratio" — fraction of trades that are self-excited vs. exogenous. High endogeneity = momentum continuation; rapid decay = exhaustion/reversal.

**Signal construction**:
```
Hawkes process: λ(t) = μ + α × sum(exp(-β(t - t_i)) for t_i < t)

Endogeneity ratio (reflexivity index):
    ρ = α / β   [must be < 1 for stationarity]
    ρ → 1.0 = self-exciting/cascading (momentum)
    ρ → 0.0 = random/exogenous (potential reversal)

Exhaustion signal:
    intensity_peak = max(λ(t)) in last 5 min
    intensity_now = λ(current_t)
    exhaustion = intensity_now < 0.5 × intensity_peak
    → exhaustion = True → reversal signal
```

**Academic basis**: Neural Hawkes (Tandfonline 2025); Filimonov & Sornette "reflexivity index" (2012). Endogeneity ratio distinguishes herding (ρ→1) from informed flow (ρ→0).

**Complexity**: HIGH — requires raw tick data (trade timestamp + direction), real-time Hawkes fitting per candle resolution. MLE fitting is O(N²) per update without FFT approximation.

**Alternative**: Use approximate Hawkes from existing `hawkes_intensity` feature (already computed in `apex_features.py`) — this is a significant simplification but loses the endogeneity ratio.

**Test on**: 5m (realtime tick data). Assets: BTC, ETH.
**Development estimate**: 2–3 weeks (tick data feed + Hawkes fitting + backtest integration).

---

### 14. Polymarket Odds Momentum Signal (`PolyOddsStrategy`)

**Core idea**: Monitor the YES/NO price on Polymarket CLOB itself as the primary signal source. Sustained directional moves in YES price before resolution reflect real-time informed flow — bet in the same direction.

**Signal construction**:
```
# Poll CLOB every 5 seconds
odds_mom = (yes_price_now - yes_price_60s_ago) / 0.5  # normalized to ±2

Entry:
    odds_mom > 0.05 (moved 5pp toward YES in 60s) → bet YES (UP)
    odds_mom < -0.05 (moved 5pp toward NO in 60s) → bet NO (DOWN)

Size decay by time-to-resolution:
    minutes_to_resolution = (next_5m_boundary - now) / 60
    size = base_size × (minutes_to_resolution / 5.0)  # smaller closer to resolution
```

**Novel angle**: Avoids Binance→Polymarket lag entirely. Uses Polymarket's own price discovery as the primary signal. Informed bettors move YES/NO price directly — this captures their activity immediately.

**Implementation**: Extends `CLOBImbalanceStrategy` pattern. Live-only (no historical odds feed for backtesting). Requires WebSocket CLOB subscription.

**Limitation**: Cannot be backtested with historical data. Paper trade only initially.
**Test on**: Real-time (sub-minute polling). Assets: All 4.

---

### 15. Bollinger Band Squeeze Breakout (`BBSqueezeStrategy`)

**Core idea**: Bollinger Band squeeze (BB width at multi-period low) followed by expansion often precedes a sustained directional move. Bet in the direction of the first breakout candle after the squeeze.

**Signal construction**:
```
bb_width = (upper_band - lower_band) / middle_band  # ≈ 4 × sigma / price

# Detect squeeze: BB width below 10th percentile of last 100 periods
squeeze = bb_width < np.percentile(bb_width[-100:], 10)

# On first candle where squeeze ends (BB width expands):
breakout_up = squeeze_{t-1} AND NOT squeeze_t AND close > upper_band
breakout_down = squeeze_{t-1} AND NOT squeeze_t AND close < lower_band

Volume confirmation:
    volume_t > 1.5 × rolling_mean(volume, 20)  # conviction filter

Signal:
    breakout_up + volume_confirm → signal = +1
    breakout_down + volume_confirm → signal = -1
```

**Academic basis**: BB squeeze documented as a precursor to trend initiation (Bollinger 2001). TTM Squeeze (standard on TradingView) built on this principle. Most reliable on 15m+ where squeezes have meaningful duration.

**Dependencies**: `bollinger()` already in `packages/indicators/`.
**Test on**: 15m, 1h (5m too noisy for meaningful squeezes). Assets: All 4.

---

## Recommended Implementation Order

### Phase 1 (This Sprint) — Gates and Filters on Existing Strategies
1. **VPIN Gate** (`VPINGate`) — minimal code, composable, high confidence
2. **Volatility-Adjusted Streak** (`AdaptiveStreakStrategy`) — small extension to existing strategy
3. **VWAP Mean Reversion** (`VWAPReversionStrategy`) — clean standalone strategy, pure OHLCV+volume

### Phase 2 (Next Sprint) — Structural Improvements
4. **Cross-TF Trend Hierarchy** (`TFHierarchyStrategy`) — operationalizes the mom_15m insight from ApexML research
5. **BTC Lead-Lag** (`BTCLeadLagStrategy`) — novel cross-asset signal
6. **Multi-Asset Streak Confluence** (`MultiAssetStreakStrategy`) — confluence filter for streaks

### Phase 3 (Research Sprint) — ML and Regime Detection
7. **HMM Regime Router** (`HMMRegimeRouter`) — requires validation of regime stability
8. **OI-CVD Divergence** (`OICVDDivergenceStrategy`) — build on existing funding/CVD data
9. **RSI Divergence Detector** (`RSIDivergenceStrategy`) — build on existing RSI indicator
10. **CatBoost ML** (`CatBoostMLStrategy`) — train after Phase 1-2 generate new feature ideas

### Phase 4 (Exploratory)
11. **Intraday Seasonal Betting** (`IntraSeasonalStrategy`) — needs 90+ days data
12. **GARCH Vol Filter** (`GARCHVolFilter`) — implement EWMA first as approximation
13. **BB Squeeze Breakout** (`BBSqueezeStrategy`) — Tier 3 candidate
14. **Polymarket Odds Momentum** (`PolyOddsStrategy`) — live-only, paper trade first
15. **Neural Hawkes** (`HawkesMMStrategy`) — long-term research track

---

## Data Requirements Checklist

| Strategy | OHLCV | CVD | Liquidations | Funding | BTC candles | Tick data | Polymarket CLOB |
|---|---|---|---|---|---|---|---|
| VPINGate | ✓ | ✓ (buy_vol/sell_vol) | — | — | — | — | — |
| HMMRegimeRouter | ✓ | — | — | — | — | — | — |
| BTCLeadLag | ✓ | — | — | — | ✓ | — | — |
| AdaptiveStreak | ✓ | — | — | — | — | — | — |
| MultiAssetStreak | ✓ (all 4) | — | — | — | — | — | — |
| TFHierarchy | ✓ (5m+15m+1h) | — | — | — | — | — | — |
| GARCHVolFilter | ✓ | — | — | — | — | — | — |
| RSIDivergence | ✓ | — | — | — | — | — | — |
| VWAPReversion | ✓ | — | — | — | — | — | — |
| IntraSeasonal | ✓ | — | — | — | — | — | — |
| OICVDDivergence | ✓ | ✓ | — | ✓ | — | — | — |
| CatBoostML | ✓ | ✓ | ✓ | ✓ | — | — | — |
| HawkesMM | — | — | — | — | — | ✓ | — |
| PolyOdds | — | — | — | — | — | — | ✓ |
| BBSqueeze | ✓ | — | — | — | — | — | — |

**All OHLCV + enrichment data available via**: `enrich_candles(candles, asset, include_cvd=True, include_liq=True, include_funding=True)`

---

## Testing Framework

### Standard Backtest Protocol (apply to all new strategies)
```bash
# 1. Walk-forward backtest (train pre-2024, test 2024+)
uv run python scripts/backtest_apex.py --strategy <name> --asset eth --tf 5m --walk-forward

# 2. Parameter sweep
uv run python scripts/sweep_gates.py --strategy <name> --asset eth --tf 5m

# 3. Multi-asset multi-TF
uv run python scripts/run_all_backtests.py --strategy <name>

# 4. Gate stacking test
uv run python scripts/backtest_apex.py --strategy streak_reversal --gate <new_gate> --asset eth --tf 5m
```

### Acceptance Criteria
A new strategy or gate passes review if it meets **all** of the following on the **test set** (2024+):

| Metric | Minimum | Target |
|---|---|---|
| Sharpe ratio | > 2.0 | > 4.0 |
| Win rate | > 52% | > 55% |
| Max drawdown | < 25% | < 15% |
| Trade count (test period) | > 100 trades | > 200 trades |
| Does not degrade existing best | Neutral | +0.3 Sharpe on ETH/5m |

### Gate Stacking Test
For new gates: stack on top of `StreakReversal + TrendFilter(ema=100)` (current best) and verify:
- Test Sharpe does not decrease by more than 0.2
- Win rate stays above 52%
- Trade count remains above 100 (gate is not too aggressive)

### Paper Trade Threshold
Any strategy with test Sharpe > 2.0 is a candidate for paper trading on VPS. Add to `docker-compose.yml` as commented service. Uncomment after 2 weeks of paper trading confirms edge.

---

## Files to Modify When Implementing

| File | Purpose |
|---|---|
| `packages/strategies/src/polymarket_algo/strategies/__init__.py` | Register new strategy classes |
| `packages/strategies/pyproject.toml` | Add new dependencies (hmmlearn, catboost) |
| `packages/data/src/polymarket_algo/data/enrich.py` | Add VWAP, VPIN columns to `enrich_candles()` |
| `packages/indicators/src/polymarket_algo/indicators/__init__.py` | Add ATR, VWAP functions |
| `scripts/backtest_apex.py` | Extend STRATEGY_MAP for new strategies |
| `scripts/sweep_gates.py` | Add new gates to GATE_MAP |
| `scripts/live_stats.py` | Add new bot history file entries |
| `docker-compose.yml` | Add commented service definitions for new bots |
