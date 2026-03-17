from __future__ import annotations

import pandas as pd


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (Wilder smoothing).

    Returns DataFrame with columns: adx, plus_di, minus_di.
    """
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    alpha = 1 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di_s = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_s
    minus_di_s = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_s

    dx = 100 * (plus_di_s - minus_di_s).abs() / (plus_di_s + minus_di_s).replace(0, float("nan"))
    adx_s = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return pd.DataFrame(
        {"adx": adx_s, "plus_di": plus_di_s, "minus_di": minus_di_s},
        index=close.index,
    )
