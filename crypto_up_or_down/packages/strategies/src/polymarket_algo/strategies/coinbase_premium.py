"""Coinbase Premium Index strategy.

The Coinbase premium index measures how much more (or less) BTC/ETH costs
on Coinbase vs Binance. A positive premium indicates US institutional/retail
demand is outpacing global prices — historically a bullish signal. A negative
premium indicates US selling pressure.

Requires candles enriched with coinbase_premium and coinbase_premium_zscore
columns. Use enrich_candles(..., include_coinbase=True) before evaluating.
"""

from __future__ import annotations

import pandas as pd


class CoinbasePremiumStrategy:
    """US demand signal via Coinbase-vs-Binance price premium z-score."""

    name = "coinbase_premium"
    description = "Bullish/bearish signal from Coinbase vs Binance price premium"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "z_thresh": 1.5,    # z-score threshold for signal
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "z_thresh": [1.0, 1.5, 2.0, 2.5],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate Coinbase premium signals.

        Args:
            candles: OHLCV DataFrame enriched with coinbase_premium_zscore column.
                     Use enrich_candles(..., include_coinbase=True).
            **params: z_thresh, size

        Returns:
            DataFrame with 'signal' (1/-1/0) and 'size' columns.
        """
        z_thresh = float(params.get("z_thresh", self.default_params["z_thresh"]))
        size_val = float(params.get("size", self.default_params["size"]))

        if "coinbase_premium_zscore" not in candles.columns:
            return pd.DataFrame({"signal": 0, "size": 0.0}, index=candles.index)

        z = candles["coinbase_premium_zscore"]
        signal = pd.Series(0, index=candles.index, dtype=int)

        # US buyers paying a premium → bullish
        signal[z > z_thresh] = 1
        # US buyers paying a discount → bearish
        signal[z < -z_thresh] = -1

        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
