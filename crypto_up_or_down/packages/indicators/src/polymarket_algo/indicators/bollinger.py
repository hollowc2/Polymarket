from __future__ import annotations

import pandas as pd


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands: middle SMA, upper, lower."""
    middle = series.rolling(window=period, min_periods=period).mean()
    rolling_std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
        },
        index=series.index,
    )
