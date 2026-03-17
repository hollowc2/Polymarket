from __future__ import annotations

import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = series.diff()
    gain = pd.Series(delta.clip(lower=0), index=series.index)
    loss = pd.Series((-delta).clip(upper=0), index=series.index)

    avg_gain = pd.Series(gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean(), index=series.index)
    avg_loss = pd.Series(loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean(), index=series.index)

    rs: pd.Series = avg_gain / avg_loss.replace(0, pd.NA)
    rsi_values: pd.Series = pd.Series(100 - (100 / (1 + rs)), index=series.index)

    both_flat = (avg_gain == 0) & (avg_loss == 0)
    rsi_values = rsi_values.where(~both_flat, 50.0)

    # Handle edge case when avg_loss is exactly zero => RSI=100
    rsi_values = rsi_values.fillna(100)
    return rsi_values
