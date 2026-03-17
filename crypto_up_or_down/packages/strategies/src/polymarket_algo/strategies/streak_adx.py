from __future__ import annotations

import pandas as pd
from polymarket_algo.indicators import adx

from ._ci_sizing import ci_size


class StreakADXStrategy:
    """Streak reversal only in choppy/ranging markets (ADX below threshold).

    Trades streak reversals exclusively when ADX indicates a non-trending
    environment, filtering out streak signals in strong trends.
    """

    name = "streak_adx"
    description = "Streak reversal in choppy regimes (ADX < threshold) with CI-based sizing"
    timeframe = "5m"

    def __init__(self, asset: str = "") -> None:
        self.asset = asset  # e.g. "BTC" — used for Wilson CI lookup

    @property
    def default_params(self) -> dict:
        return {
            "trigger": 4,
            "adx_period": 14,
            "adx_threshold": 20,
            "size": 15.0,
            "use_ci_sizing": True,
        }

    @property
    def param_grid(self) -> dict[str, list]:
        return {
            "trigger": [3, 4, 5],
            "adx_period": [10, 14],
            "adx_threshold": [15, 20, 25],
            "size": [15.0],
            "use_ci_sizing": [True, False],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        trigger = int(params.get("trigger", 4))
        adx_period = int(params.get("adx_period", 14))
        adx_threshold = float(params.get("adx_threshold", 20))
        size_val = float(params.get("size", 15.0))
        use_ci = bool(params.get("use_ci_sizing", True))

        adx_df = adx(candles["high"], candles["low"], candles["close"], period=adx_period)
        is_choppy = adx_df["adx"] < adx_threshold

        direction = (candles["close"].diff() > 0).map({True: 1, False: -1}).fillna(0)
        streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1

        signal = pd.Series(0, index=candles.index, dtype=int)
        signal[(streak >= trigger) & (direction == 1) & is_choppy] = -1
        signal[(streak >= trigger) & (direction == -1) & is_choppy] = 1

        size = ci_size(signal, streak, size_val, use_ci, self.timeframe, self.asset)
        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
