from __future__ import annotations

from typing import Any

import pandas as pd
from polymarket_algo.indicators import bollinger_bands


class BollingerSqueezeStrategy:
    name = "bollinger_squeeze"
    description = "Buy/sell breakout after Bollinger Band squeeze period"
    timeframe = "5m"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "period": 20,
            "std_dev": 2.0,
            "squeeze_lookback": 20,
            "squeeze_pct": 0.20,
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "period": [14, 20, 26],
            "std_dev": [1.5, 2.0, 2.5],
            "squeeze_lookback": [15, 20],
            "squeeze_pct": [0.15, 0.20, 0.25],
            "size": [15.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params: Any) -> pd.DataFrame:
        config = {**self.default_params, **params}
        period = int(config["period"])
        std_dev = float(config["std_dev"])
        squeeze_lookback = int(config["squeeze_lookback"])
        squeeze_pct = float(config["squeeze_pct"])
        size_val = float(config["size"])

        bands = bollinger_bands(candles["close"], period=period, std_dev=std_dev)
        band_width = (bands["upper"] - bands["lower"]) / bands["middle"]

        # Squeeze = current band_width is in the bottom squeeze_pct of last squeeze_lookback bars
        squeeze = band_width < band_width.rolling(squeeze_lookback).quantile(squeeze_pct)
        was_squeezing = squeeze.shift(1)

        breakout_up = was_squeezing & (candles["close"] > bands["upper"])
        breakout_down = was_squeezing & (candles["close"] < bands["lower"])

        signal = breakout_up.astype(int) - breakout_down.astype(int)
        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
