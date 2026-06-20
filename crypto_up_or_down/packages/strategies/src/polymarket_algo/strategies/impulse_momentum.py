from __future__ import annotations

from typing import Any

import pandas as pd


class ImpulseMomentumStrategy:
    """Follow a large BTC move when Polymarket pricing confirms the direction."""

    name = "impulse_momentum"
    description = "BTC 5m interval impulse confirmed by matching Polymarket CLOB skew"
    timeframe = "5m"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "impulse_usd_min": 70.0,
            "threshold_price": 0.70,
            "size": 5.0,
        }

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "impulse_usd_min": [50.0, 70.0, 100.0],
            "threshold_price": [0.65, 0.70, 0.75],
            "size": [2.0, 5.0, 8.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Return a signal only when spot impulse and CLOB crowd skew agree.

        Required columns are ``open``, ``close``, ``up_ask``, and ``down_ask``.
        The advertised $70-$100 impulse is treated as a minimum confirmation,
        not a maximum; larger moves still qualify.
        """
        config = {**self.default_params, **params}
        impulse_min = float(config["impulse_usd_min"])
        threshold = float(config["threshold_price"])
        size_value = float(config["size"])

        missing = {"open", "close", "up_ask", "down_ask"} - set(candles.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        impulse = candles["close"].astype(float) - candles["open"].astype(float)
        up_ask = candles["up_ask"].astype(float)
        down_ask = candles["down_ask"].astype(float)

        bullish = (impulse >= impulse_min) & (up_ask >= threshold) & (up_ask > down_ask)
        bearish = (impulse <= -impulse_min) & (down_ask >= threshold) & (down_ask > up_ask)

        signal = bullish.astype(int) - bearish.astype(int)
        selected_ask = pd.Series(float("nan"), index=candles.index, dtype=float)
        selected_ask.loc[bullish] = up_ask.loc[bullish]
        selected_ask.loc[bearish] = down_ask.loc[bearish]

        size = pd.Series(0.0, index=candles.index)
        size.loc[signal != 0] = size_value

        return pd.DataFrame(
            {
                "signal": signal,
                "size": size,
                "impulse_usd": impulse,
                "selected_ask": selected_ask,
            },
            index=candles.index,
        )
