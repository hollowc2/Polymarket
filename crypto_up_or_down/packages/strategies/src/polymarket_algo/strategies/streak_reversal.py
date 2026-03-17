import pandas as pd


class StreakReversalStrategy:
    name = "streak_reversal"
    description = "Reversal on directional candle streak"
    timeframe = "5m"

    @property
    def default_params(self):
        return {"trigger": 4, "size": 15.0}

    @property
    def param_grid(self):
        return {"trigger": [3, 4, 5], "size": [10.0, 15.0, 20.0]}

    def evaluate(self, candles: pd.DataFrame, **params):
        trigger = int(params.get("trigger", 4))
        size_val = float(params.get("size", 15.0))
        direction = (candles["close"].diff() > 0).map({True: 1, False: -1}).fillna(0)
        streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
        signal = pd.Series(0, index=candles.index, dtype=int)
        signal[(streak >= trigger) & (direction == 1)] = -1
        signal[(streak >= trigger) & (direction == -1)] = 1
        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0
        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
