"""Shared Wilson CI half-Kelly sizing helper for streak strategies."""

from __future__ import annotations

import pandas as pd
from polymarket_algo.core.sizing import get_rate_estimate

# Polymarket payout: win ~$0.45 on $0.50 risk → b ≈ 0.9
_B = 0.9
# BTC 5m trigger=4 half-Kelly reference (pre-computed)
_F_REF = 0.014


def ci_size(
    signal: pd.Series,
    streak: pd.Series,
    base_size: float,
    use_ci: bool,
    timeframe: str,
    asset: str,
) -> pd.Series:
    """Size each trade by Wilson CI half-Kelly. Falls back to flat if no CI data.

    Args:
        signal:    signal series (1/-1/0)
        streak:    streak length at each bar
        base_size: flat USD amount used when use_ci=False or no CI data
        use_ci:    whether to use CI-based sizing
        timeframe: e.g. "5m"
        asset:     e.g. "BTC" — looked up in ASSET_REVERSAL_RATES
    """
    size = pd.Series(0.0, index=signal.index)
    active = signal[signal != 0].index

    if not use_ci:
        size.loc[active] = base_size
        return size

    for idx in active:
        sl = int(streak.loc[idx])
        est = get_rate_estimate(timeframe, sl, asset.upper())
        if est is not None and est.ci_lo > 0.50:
            f_half = 0.5 * max((est.rate * _B - (1 - est.rate)) / _B, 0.0)
            size.loc[idx] = base_size * f_half / _F_REF
        else:
            size.loc[idx] = base_size  # no CI data → flat fallback
    return size
