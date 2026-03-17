"""APEX Feature Engine — microstructure signal computation.

Computes order-flow and price microstructure features from enriched candle
DataFrames produced by enrich_candles() (polymarket_algo.data).

All features degrade gracefully when enriched columns are absent so the
strategy can also run on plain OHLCV data (lower signal quality).

Feature summary
---------------
TFI              Trade Flow Imbalance = (buy - sell) / total ∈ [-1, 1]
OBI proxy        Rolling TFI deviation — approximates order-book imbalance ∈ [-1, 1]
microprice       OHLCV volume-weighted synthetic mid price
microprice_drift microprice − midprice, ATR-normalised ∈ [-3, 3]
hawkes_intensity Self-exciting Hawkes intensity from volume events ∈ [0, 1]
cascade_score    Liquidation cascade severity (signed) ∈ [-∞, ∞]
funding_pressure Funding z-score × price momentum × OI proxy ∈ [-1, 1]

Live-only upgrade (pass a CachedOrderBook from executor.ws):
    compute_obi_from_book(bid_vol, ask_vol) → true scalar OBI
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Trade Flow Imbalance (TFI)
# ---------------------------------------------------------------------------


def compute_tfi(candles: pd.DataFrame) -> pd.Series:
    """Trade Flow Imbalance from Binance aggTrade CVD columns.

    TFI = (buy_vol - sell_vol) / (buy_vol + sell_vol)

    Returns a zero-filled series when CVD columns are unavailable.
    """
    if "buy_vol" not in candles.columns or "sell_vol" not in candles.columns:
        return pd.Series(0.0, index=candles.index, name="tfi")

    total = candles["buy_vol"] + candles["sell_vol"]
    tfi = (candles["buy_vol"] - candles["sell_vol"]) / total.replace(0.0, np.nan)
    return tfi.fillna(0.0).rename("tfi")


# ---------------------------------------------------------------------------
# Order Book Imbalance (OBI) — proxy and true live variant
# ---------------------------------------------------------------------------


def compute_obi_proxy(candles: pd.DataFrame, window: int = 12) -> pd.Series:
    """Order Book Imbalance proxy via rolling TFI z-score.

    In the absence of live order-book snapshots we approximate OBI as the
    z-score of TFI vs its rolling mean, clipped to [-1, 1].

    Positive → accumulated buy-side pressure.
    Negative → accumulated sell-side pressure.
    """
    tfi = compute_tfi(candles)
    roll_mean = tfi.rolling(window, min_periods=1).mean()
    roll_std = tfi.rolling(window, min_periods=1).std().replace(0.0, np.nan)
    obi = ((tfi - roll_mean) / roll_std).fillna(0.0)
    return obi.clip(-1.0, 1.0).rename("obi_proxy")


def compute_obi_from_book(bid_volume: float, ask_volume: float) -> float:
    """True OBI from a live Polymarket order book.

    OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

    Use with CachedOrderBook.get_execution_price() data from executor.ws
    during live operation.
    """
    total = bid_volume + ask_volume
    if total == 0.0:
        return 0.0
    return (bid_volume - ask_volume) / total


# ---------------------------------------------------------------------------
# Microprice and Microprice Drift
# ---------------------------------------------------------------------------


def compute_microprice(candles: pd.DataFrame) -> pd.Series:
    """Volume-weighted synthetic microprice from OHLCV (+ CVD when available).

    True microprice (requires live book):
        mp = (ask × bid_size + bid × ask_size) / (bid_size + ask_size)

    Backtest-compatible approximation using CVD:
        mp ≈ (high × sell_vol + low × buy_vol) / (buy_vol + sell_vol)

    Intuition: buyers lift the ask (push toward the high); sellers hit the
    bid (push toward the low).  Volume-weighting of high/low encodes this
    directional pressure without needing live book data.

    Falls back to (H + L + C) / 3 (typical price) when CVD is unavailable.
    """
    if "buy_vol" in candles.columns and "sell_vol" in candles.columns:
        total = candles["buy_vol"] + candles["sell_vol"]
        mp = (candles["high"] * candles["sell_vol"] + candles["low"] * candles["buy_vol"]) / total.replace(0.0, np.nan)
        mp = mp.fillna((candles["high"] + candles["low"]) / 2.0)
    else:
        mp = (candles["high"] + candles["low"] + candles["close"]) / 3.0
    return mp.rename("microprice")


def compute_microprice_drift(candles: pd.DataFrame) -> pd.Series:
    """ATR-normalised microprice drift relative to the simple midprice.

    drift = (microprice - midprice) / ATR_20

    Positive drift → microprice > midprice → net buy-side pressure.
    Negative drift → microprice < midprice → net sell-side pressure.
    Clipped to [-3, 3] for compatibility with sigmoid-based edge models.
    """
    mp = compute_microprice(candles)
    midprice = (candles["high"] + candles["low"]) / 2.0
    raw_drift = mp - midprice
    atr = (candles["high"] - candles["low"]).rolling(20, min_periods=1).mean()
    normalised = (raw_drift / atr.replace(0.0, np.nan)).fillna(0.0)
    return normalised.clip(-3.0, 3.0).rename("microprice_drift")


# ---------------------------------------------------------------------------
# Hawkes Process Intensity
# ---------------------------------------------------------------------------


def compute_hawkes_intensity(
    candles: pd.DataFrame,
    mu: float = 0.10,
    alpha: float = 0.30,
    beta: float = 1.00,
) -> pd.Series:
    """Self-exciting Hawkes process intensity estimated at each candle.

    Hawkes model (Ogata 1988):
        λ_t = μ + Σ_i α · exp(−β · (t − t_i))

    Closed-form candle recursion:
        λ_t = μ + (λ_{t−1} − μ) · e^{−β} + α · event_{t−1}

    Event magnitude:
        |delta| when CVD columns are available (trade-flow intensity).
        volume  as a fallback proxy.

    The event magnitude is percentile-ranked over a 50-bar rolling window
    so that the intensity is comparable across assets and time periods.

    Output is passed through sigmoid to produce a value in (0, 1).

    Args:
        mu:    Background (baseline) arrival rate.
        alpha: Self-excitation coefficient — how much each event boosts λ.
        beta:  Memory decay rate (candles); β = 1.0 ≈ 63 % decay per bar.
    """
    if "delta" in candles.columns:
        events = candles["delta"].abs()
    else:
        events = candles["volume"]

    # Normalise events to [0, 1] via rolling percentile rank
    event_rank = events.rolling(50, min_periods=5).rank(pct=True).fillna(0.5)

    n = len(candles)
    intensity = np.empty(n)
    if n == 0:
        return pd.Series([], dtype=float, name="hawkes_intensity")

    intensity[0] = mu
    decay = float(np.exp(-beta))

    for i in range(1, n):
        intensity[i] = mu + (intensity[i - 1] - mu) * decay + alpha * float(event_rank.iloc[i - 1])

    raw = pd.Series(intensity, index=candles.index)
    # Sigmoid maps intensity → (0, 1); values above ~3 saturate at ≈0.95
    result = 1.0 / (1.0 + np.exp(-raw.clip(-10, 10)))
    return result.rename("hawkes_intensity")


# ---------------------------------------------------------------------------
# Liquidation Cascade Score
# ---------------------------------------------------------------------------


def compute_cascade_score(
    candles: pd.DataFrame,
    liq_lookback: int = 30,
) -> pd.Series:
    """Signed liquidation cascade severity.

    cascade_score = 2.0 × liq_imbalance
                  + 1.5 × oi_shock
                  + 1.0 × proximity

    Components
    ----------
    liq_imbalance  (liq_long - liq_short) / total_liq
                   > 0 → more longs liquidated → bearish cascade
                   < 0 → more shorts liquidated → bullish cascade (squeeze)
    oi_shock       rolling z-score of total liquidation USD (how extreme)
    proximity      intrabar body / bar range (how decisive the move was)

    Returns zeros when liquidation columns are unavailable.
    """
    if "liq_long_usd" not in candles.columns or "liq_short_usd" not in candles.columns:
        return pd.Series(0.0, index=candles.index, name="cascade_score")

    liq_long = candles["liq_long_usd"].fillna(0.0)
    liq_short = candles["liq_short_usd"].fillna(0.0)
    total_liq = liq_long + liq_short

    # Signed liquidation imbalance
    denom = total_liq.replace(0.0, np.nan)
    liq_imbalance = ((liq_long - liq_short) / denom).fillna(0.0)

    # OI shock: rolling z-score of total liquidation
    liq_mean = total_liq.rolling(liq_lookback, min_periods=5).mean()
    liq_std = total_liq.rolling(liq_lookback, min_periods=5).std().replace(0.0, np.nan)
    oi_shock = ((total_liq - liq_mean) / liq_std).fillna(0.0).clip(-3.0, 3.0) / 3.0

    # Proximity: ratio of candle body to bar range
    bar_range = (candles["high"] - candles["low"]).replace(0.0, np.nan)
    body = (candles["close"] - candles["open"]).abs()
    proximity = (body / bar_range).fillna(0.0).clip(0.0, 1.0)

    score = 2.0 * liq_imbalance + 1.5 * oi_shock + 1.0 * proximity
    return score.rename("cascade_score")


# ---------------------------------------------------------------------------
# Funding Pressure
# ---------------------------------------------------------------------------


def compute_funding_pressure(
    candles: pd.DataFrame,
    mom_window: int = 5,
) -> pd.Series:
    """Funding rate × price momentum × OI proxy pressure signal.

    funding_pressure = funding_z × price_momentum × oi_growth

    Each component is normalised to [-1, 1] before multiplication.

    funding_z       funding_zscore column from enrich_candles()
    price_momentum  N-bar return z-scored over 50 bars
    oi_growth       volume z-score (proxy for open interest change)

    Positive pressure → crowded longs + rising price + rising OI → fade risk.
    Negative pressure → crowded shorts + falling price + rising OI → squeeze risk.

    Returns zeros when funding columns are unavailable.
    """
    if "funding_zscore" not in candles.columns:
        return pd.Series(0.0, index=candles.index, name="funding_pressure")

    funding_z = candles["funding_zscore"].fillna(0.0).clip(-4.0, 4.0) / 4.0

    # Price momentum
    returns = candles["close"].pct_change(mom_window).fillna(0.0)
    ret_std = returns.rolling(50, min_periods=5).std().replace(0.0, np.nan)
    price_mom = (returns / ret_std).fillna(0.0).clip(-2.0, 2.0) / 2.0

    # OI growth proxy via volume z-score
    vol_mean = candles["volume"].rolling(50, min_periods=5).mean()
    vol_std = candles["volume"].rolling(50, min_periods=5).std().replace(0.0, np.nan)
    oi_growth = ((candles["volume"] - vol_mean) / vol_std).fillna(0.0).clip(-2.0, 2.0) / 2.0

    pressure = funding_z * price_mom * oi_growth
    return pressure.clip(-1.0, 1.0).rename("funding_pressure")
