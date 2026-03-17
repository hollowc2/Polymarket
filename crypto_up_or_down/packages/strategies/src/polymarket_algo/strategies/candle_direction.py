from __future__ import annotations

from typing import Any, cast

import pandas as pd
from polymarket_algo.indicators import ema, macd, rsi


class CandleDirectionStrategy:
    name = "candle_direction"
    description = "EMA/MACD/RSI alignment strategy with optional stronger position sizing"
    timeframe = "5m"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_period": 14,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        }

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "ema_fast": [8, 12, 16],
            "ema_slow": [21, 26, 34],
            "rsi_period": [10, 14, 21],
            "rsi_overbought": [65.0, 70.0, 75.0],
            "rsi_oversold": [25.0, 30.0, 35.0],
            "macd_fast": [8, 12],
            "macd_slow": [21, 26],
            "macd_signal": [7, 9],
        }

    def evaluate(self, candles: pd.DataFrame, **params: Any) -> pd.DataFrame:
        config = {**self.default_params, **params}

        close = cast(pd.Series, candles["close"])

        ema_fast_line = ema(close, int(config["ema_fast"]))
        ema_slow_line = ema(close, int(config["ema_slow"]))
        macd_df = macd(
            close,
            fast_period=int(config["macd_fast"]),
            slow_period=int(config["macd_slow"]),
            signal_period=int(config["macd_signal"]),
        )
        rsi_values = rsi(close, period=int(config["rsi_period"]))

        bullish_ema = ema_fast_line > ema_slow_line
        bearish_ema = ema_fast_line < ema_slow_line

        bullish_macd = macd_df["macd"] > macd_df["signal"]
        bearish_macd = macd_df["macd"] < macd_df["signal"]

        bullish_rsi = rsi_values > float(config["rsi_oversold"])
        bearish_rsi = rsi_values < float(config["rsi_overbought"])

        long_cond = bullish_ema & bullish_macd & bullish_rsi
        short_cond = bearish_ema & bearish_macd & bearish_rsi

        signal = pd.Series(0, index=candles.index, dtype=int)
        signal.loc[long_cond] = 1
        signal.loc[short_cond] = -1

        strong_long = long_cond & (macd_df["histogram"] > 0) & (rsi_values.between(50, 65))
        strong_short = short_cond & (macd_df["histogram"] < 0) & (rsi_values.between(35, 50))

        size = pd.Series(15.0, index=candles.index)
        size.loc[(strong_long | strong_short) & (signal != 0)] = 20.0
        size.loc[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
