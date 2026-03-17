"""APEX Filters — adverse selection, regime detection, and queue position.

These filters protect the APEX strategy from entering trades in unfavourable
market microstructure conditions.

Classes / functions
-------------------
compute_adverse_selection  Toxicity score for order flow quality ∈ [0, 1]
detect_regime              Boolean mask — True = conditions ok to trade
estimate_fill_probability  Queue position model for passive limit orders
_rolling_zscore            Internal utility
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .apex_features import compute_tfi  # noqa: F401 — re-exported for convenience

# ---------------------------------------------------------------------------
# Adverse Selection Filter
# ---------------------------------------------------------------------------


def compute_adverse_selection(
    candles: pd.DataFrame,
    intensity_window: int = 10,
    spread_window: int = 20,
) -> pd.Series:
    """Estimate toxic order-flow probability for each candle.

    A high toxicity score indicates that the current flow is likely driven by
    informed traders (adverse selection), stale quotes, or market manipulation
    — conditions where passive limit orders are likely to be picked off.

    Components (each normalised to [0, 1]):
        price_impact   |close − midprice| / midprice — z-scored
        trade_intensity |delta| / volume (order-flow dominance) — z-scored
        spread_widening (H − L) / close vs rolling mean — z-scored
        depth_shock    volume spike followed by a price reversal

    Weights: 35 / 30 / 25 / 10 (sum = 1.0)

    Returns toxicity ∈ [0, 1]; values above 0.6 are considered high-toxicity.
    """
    midprice = (candles["high"] + candles["low"]) / 2.0

    # --- Price impact ---
    price_impact = ((candles["close"] - midprice).abs() / midprice.replace(0.0, np.nan)).fillna(0.0)
    impact_z = _rolling_zscore(price_impact, spread_window).clip(0.0, 3.0) / 3.0

    # --- Trade intensity from CVD ---
    if "delta" in candles.columns and "volume" in candles.columns:
        intensity = (candles["delta"].abs() / candles["volume"].replace(0.0, np.nan)).fillna(0.0)
        intensity_z = _rolling_zscore(intensity, intensity_window).clip(0.0, 3.0) / 3.0
    else:
        intensity_z = pd.Series(0.0, index=candles.index)

    # --- Spread proxy (H-L range as fraction of close) ---
    hl_spread = ((candles["high"] - candles["low"]) / candles["close"].replace(0.0, np.nan)).fillna(0.0)
    spread_z = _rolling_zscore(hl_spread, spread_window).clip(0.0, 3.0) / 3.0

    # --- Depth shock: large-volume bar followed by price reversal ---
    vol_z = _rolling_zscore(candles["volume"], spread_window)
    close_diff = candles["close"].diff()
    price_dir = close_diff.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    prev_vol_z = vol_z.shift(1).fillna(0.0)
    prev_dir = price_dir.shift(1).fillna(0)
    depth_shock = ((prev_vol_z > 2.0) & (price_dir != 0) & (price_dir != prev_dir)).astype(float)

    toxicity = 0.35 * impact_z + 0.30 * intensity_z + 0.25 * spread_z + 0.10 * depth_shock
    return toxicity.clip(0.0, 1.0).rename("toxicity")


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------


def detect_regime(
    candles: pd.DataFrame,
    vol_window: int = 20,
    rank_window: int = 200,
    min_vol_pct: float = 0.25,
    max_vol_pct: float = 0.85,
    min_volume_pct: float = 0.20,
) -> pd.Series:
    """Market regime filter — True = conditions suitable for APEX trading.

    Two conditions must both hold:

    1. Realised volatility (ATR) is in [min_vol_pct, max_vol_pct] percentile.
       Too quiet  → no exploitable microstructure edge.
       Too violent → slippage and gap risk exceed expected edge.

    2. Volume is above the min_volume_pct percentile.
       Low volume → wide spreads, thin books, poor fill quality.

    Args:
        vol_window:    ATR smoothing window (candles).
        rank_window:   Lookback for percentile ranking.
        min_vol_pct:   Minimum ATR percentile (0–1).
        max_vol_pct:   Maximum ATR percentile (0–1).
        min_volume_pct Minimum volume percentile (0–1).

    Returns:
        Boolean Series; True where both conditions are met.
    """
    atr = (candles["high"] - candles["low"]).rolling(vol_window, min_periods=5).mean()
    atr_pct = atr.rolling(rank_window, min_periods=10).rank(pct=True).fillna(0.5)

    vol_pct = candles["volume"].rolling(rank_window, min_periods=10).rank(pct=True).fillna(0.5)

    vol_ok = (atr_pct >= min_vol_pct) & (atr_pct <= max_vol_pct)
    liq_ok = vol_pct >= min_volume_pct

    return (vol_ok & liq_ok).rename("regime_ok")


# ---------------------------------------------------------------------------
# Queue Position / Fill Probability Model
# ---------------------------------------------------------------------------


def estimate_fill_probability(
    queue_position: float,
    trade_flow_rate: float,
    time_horizon: float = 60.0,
) -> float:
    """Estimate passive limit-order fill probability from queue position.

    fill_probability = (trade_flow_rate × time_horizon) / queue_position

    Intuition: if flow_rate shares/second pass through this price level and
    we have queue_position shares ahead of us, expected time to fill is
    queue_position / flow_rate seconds.  Capped at 1.0.

    Args:
        queue_position:  Volume (base units) ahead of our order at that price.
        trade_flow_rate: Average volume per second at this level.
        time_horizon:    Seconds before we intend to cancel unfilled orders.

    Returns:
        Estimated fill probability ∈ [0.0, 1.0].
        Returns 1.0 when queue_position ≤ 0 (we are at the front).
    """
    if queue_position <= 0.0:
        return 1.0
    raw = trade_flow_rate * time_horizon / queue_position
    return float(np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with fillna(0) for NaN-safe computation."""
    mean = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std().replace(0.0, np.nan)
    return ((series - mean) / std).fillna(0.0)
