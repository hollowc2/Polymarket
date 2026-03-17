from __future__ import annotations

from typing import Any

import pandas as pd
from polymarket_algo.indicators.hl_orderflow import hl_orderflow_signal


class PinBarReversalStrategy:
    name = "pin_bar_reversal"
    description = "Reversal on pin bar (long wick, small body) candle pattern"
    timeframe = "5m"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "body_threshold": 0.35,
            "wick_threshold": 0.55,
            "size": 15.0,
            "hl_gate": False,  # veto reversals that oppose HL 5m flow
            "hl_coin": "BTC",  # coin to check for HL gate
        }

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "body_threshold": [0.25, 0.35, 0.45],
            "wick_threshold": [0.50, 0.55, 0.60],
            "size": [15.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params: Any) -> pd.DataFrame:
        config = {**self.default_params, **params}
        body_threshold = float(config["body_threshold"])
        wick_threshold = float(config["wick_threshold"])
        size_val = float(config["size"])

        total_range = (candles["high"] - candles["low"]).replace(0, float("nan"))
        body = (candles["close"] - candles["open"]).abs()
        upper_wick = candles["high"] - candles[["close", "open"]].max(axis=1)
        lower_wick = candles[["close", "open"]].min(axis=1) - candles["low"]

        body_ratio = body / total_range
        bullish_pin = (
            (body_ratio < body_threshold) & (lower_wick / total_range > wick_threshold) & (lower_wick > upper_wick)
        )
        bearish_pin = (
            (body_ratio < body_threshold) & (upper_wick / total_range > wick_threshold) & (upper_wick > lower_wick)
        )

        signal = bullish_pin.astype(int) - bearish_pin.astype(int)
        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        out = pd.DataFrame({"signal": signal, "size": size}, index=candles.index)

        if config.get("hl_gate"):
            coin = str(config.get("hl_coin", "BTC"))
            sig_5m = hl_orderflow_signal(coin, "5m")
            # Veto bullish pin bars when HL says SELL — reversal into sell pressure
            if sig_5m == "SELL":
                out.loc[out["signal"] == 1, ["signal", "size"]] = [0, 0.0]
            # Veto bearish pin bars when HL says BUY — reversal into buy pressure
            elif sig_5m == "BUY":
                out.loc[out["signal"] == -1, ["signal", "size"]] = [0, 0.0]

        return out
