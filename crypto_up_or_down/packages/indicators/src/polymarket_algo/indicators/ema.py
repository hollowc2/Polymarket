from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential moving average."""
    result = series.ewm(span=period, adjust=False, min_periods=period).mean()
    return pd.Series(result, index=series.index)
