from __future__ import annotations

import pandas as pd

from .ema import ema


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """MACD line, signal, and histogram."""
    macd_line = ema(series, fast_period) - ema(series, slow_period)
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        },
        index=series.index,
    )
