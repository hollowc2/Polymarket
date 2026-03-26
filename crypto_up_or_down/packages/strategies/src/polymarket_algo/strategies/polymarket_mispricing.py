"""Polymarket Fair Value Mispricing strategy.

Computes a theoretical fair value P(close > open) from Binance OHLCV history
and compares it to the live Polymarket YES token price. When the market price
deviates significantly from the fair value, a trade is warranted.

Fair value estimation:
  - Historical base rate: rolling fraction of 5m candles where close > open
    over a long window (500 bars ≈ 1.7 days)
  - Recent regime weight: blend in recent N-bar win rate to capture momentum
  - fair_value = (1 - w) * base_rate + w * recent_rate

Signal:
  YES_price < fair_value - thresh  → YES is cheap → bet UP
  YES_price > fair_value + thresh  → YES is expensive → bet DOWN

Live-only strategy: bot injects 'yes_price' via params at runtime.

Backtest mode: when yes_price is not provided, the strategy uses fair_value vs 0.5
as a weaker approximation. This can validate the base rate signal but will not
reflect actual Polymarket pricing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PolymarketMispricingStrategy:
    """Trade when Polymarket YES price deviates from Binance-implied fair value."""

    name = "polymarket_mispricing"
    description = "Fair-value vs live YES price mispricing — buy cheap, sell expensive"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "lookback": 500,          # historical base rate window (bars)
            "recent_window": 20,       # recent regime window (bars)
            "recent_weight": 0.3,      # weight of recent regime [0, 1]
            "misprice_thresh": 0.04,   # minimum gap to trade (4 cents)
            "yes_price": None,         # injected by bot at runtime (float 0–1)
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "lookback": [300, 500, 750],
            "recent_window": [10, 20, 30],
            "recent_weight": [0.2, 0.3, 0.5],
            "misprice_thresh": [0.03, 0.04, 0.05],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate Polymarket YES price mispricing.

        Args:
            candles: OHLCV DataFrame indexed by open_time. Must contain
                     'open' and 'close' columns.
            **params:
                lookback: long-window bars for historical base rate
                recent_window: short-window bars for recent regime
                recent_weight: blend weight for recent regime
                misprice_thresh: minimum |gap| to trigger a trade
                yes_price: current Polymarket YES price (0–1); None = backtest mode
                size: bet size in USD

        Returns:
            DataFrame with 'signal' (1/-1/0) and 'size' columns.
        """
        lookback = int(params.get("lookback", self.default_params["lookback"]))
        recent_window = int(params.get("recent_window", self.default_params["recent_window"]))
        recent_weight = float(params.get("recent_weight", self.default_params["recent_weight"]))
        misprice_thresh = float(params.get("misprice_thresh", self.default_params["misprice_thresh"]))
        yes_price = params.get("yes_price")
        size_val = float(params.get("size", self.default_params["size"]))

        signal = pd.Series(0, index=candles.index, dtype=int)
        size = pd.Series(0.0, index=candles.index)

        # Historical base rate: P(close > open) over rolling long window
        is_up = (candles["close"] > candles["open"]).astype(float)
        base_rate = is_up.rolling(lookback, min_periods=50).mean()
        recent_rate = is_up.rolling(recent_window, min_periods=5).mean()

        # Blend into fair value
        fair_value = (1.0 - recent_weight) * base_rate + recent_weight * recent_rate

        if yes_price is not None:
            # Live mode: evaluate only last bar
            last_fv = fair_value.iloc[-1]
            if np.isnan(last_fv):
                return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)

            gap = last_fv - float(yes_price)
            if gap > misprice_thresh:
                # YES is cheap relative to fair value → bet UP
                signal.iloc[-1] = 1
                size.iloc[-1] = size_val
            elif gap < -misprice_thresh:
                # YES is expensive (NO is cheap) → bet DOWN
                signal.iloc[-1] = -1
                size.iloc[-1] = size_val
        else:
            # Backtest mode: compare fair_value vs 0.5 (no actual market prices)
            gap = fair_value - 0.5
            signal[gap > misprice_thresh] = 1
            signal[gap < -misprice_thresh] = -1
            size[signal != 0] = size_val

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
