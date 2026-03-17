from __future__ import annotations

import pandas as pd
from polymarket_algo.indicators import rsi

from ._ci_sizing import ci_size


class StreakRSIStrategy:
    """Streak reversal confirmed by RSI extreme.

    Long on down-streak when RSI is oversold; short on up-streak when overbought.
    Cuts mid-range noise by requiring RSI confirmation.
    """

    name = "streak_rsi"
    description = "Streak reversal + RSI extreme confirmation with CI-based sizing"
    timeframe = "5m"

    def __init__(self, asset: str = "") -> None:
        self.asset = asset  # e.g. "BTC" — used for Wilson CI lookup

    @property
    def default_params(self) -> dict:
        return {
            "trigger": 4,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "size": 15.0,
            "use_ci_sizing": True,
        }

    @property
    def param_grid(self) -> dict[str, list]:
        return {
            "trigger": [3, 4, 5],
            "rsi_period": [10, 14],
            "rsi_overbought": [65, 70, 75],
            "size": [15.0],
            "use_ci_sizing": [True, False],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        trigger = int(params.get("trigger", 4))
        rsi_period = int(params.get("rsi_period", 14))
        rsi_overbought = float(params.get("rsi_overbought", 70))
        size_val = float(params.get("size", 15.0))
        use_ci = bool(params.get("use_ci_sizing", True))

        direction = (candles["close"].diff() > 0).map({True: 1, False: -1}).fillna(0)
        streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1

        rsi_vals = rsi(candles["close"], period=rsi_period)
        overbought = rsi_vals > rsi_overbought
        oversold = rsi_vals < (100 - rsi_overbought)

        signal = pd.Series(0, index=candles.index, dtype=int)
        signal[(streak >= trigger) & (direction == 1) & overbought] = -1
        signal[(streak >= trigger) & (direction == -1) & oversold] = 1

        size = ci_size(signal, streak, size_val, use_ci, self.timeframe, self.asset)
        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
