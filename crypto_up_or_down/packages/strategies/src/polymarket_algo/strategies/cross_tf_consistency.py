"""Cross-Timeframe Consistency strategy.

Polymarket simultaneously offers 5m, 15m, and 60m UP/DOWN markets for BTC/ETH.
These markets should be roughly consistent in their implied probabilities. When
they diverge significantly, the mispriced shorter-timeframe market is tradeable.

Signal logic:
  - 60m YES price represents P(BTC close_60m > open_60m)
  - 5m YES price represents P(BTC close_5m > open_5m) for the CURRENT interval

  If P(60m UP) >> P(5m UP) for the current interval → 5m seems underpriced → bet UP
  If P(60m DOWN) >> P(5m DOWN)                       → 5m seems overpriced → bet DOWN

The threshold accounts for the fact that 60m and 5m outcomes are not independent
(they are functions of the same underlying price path).

LIVE-ONLY strategy. Bot injects runtime params:
  - yes_price_5m:  current 5m YES token price
  - yes_price_60m: current 60m YES token price
  - yes_price_15m: (optional) current 15m YES token price for additional confirmation

This is a purely Polymarket-native signal — no Binance data required.
"""

from __future__ import annotations

import pandas as pd


class CrossTFConsistencyStrategy:
    """Trade mispriced 5m markets when 5m/60m Polymarket prices are inconsistent."""

    name = "cross_tf_consistency"
    description = "Trade when 5m and 60m Polymarket implied probabilities diverge"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "consistency_thresh": 0.08,  # minimum |gap| between TF probabilities
            "require_15m_confirm": False, # also check 15m alignment
            # Runtime params (injected by bot):
            "yes_price_5m": None,
            "yes_price_60m": None,
            "yes_price_15m": None,
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "consistency_thresh": [0.05, 0.08, 0.10, 0.12],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate cross-timeframe Polymarket probability consistency.

        Args:
            candles: OHLCV DataFrame (used only for index; candle content ignored).
            **params:
                consistency_thresh: gap threshold to trigger a trade
                require_15m_confirm: if True, 15m must also align with 60m
                yes_price_5m: Polymarket 5m YES price (injected by bot)
                yes_price_60m: Polymarket 60m YES price (injected by bot)
                yes_price_15m: Polymarket 15m YES price (optional)
                size: bet size in USD

        Returns:
            DataFrame with 'signal' (1/-1/0) and 'size' columns.
            Signal is only set on the last row.
        """
        consistency_thresh = float(params.get("consistency_thresh", self.default_params["consistency_thresh"]))
        require_15m = bool(params.get("require_15m_confirm", self.default_params["require_15m_confirm"]))
        yes_5m = params.get("yes_price_5m")
        yes_60m = params.get("yes_price_60m")
        yes_15m = params.get("yes_price_15m")
        size_val = float(params.get("size", self.default_params["size"]))

        signal = pd.Series(0, index=candles.index, dtype=int)
        size = pd.Series(0.0, index=candles.index)

        if yes_5m is None or yes_60m is None:
            return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)

        p5m = float(yes_5m)
        p60m = float(yes_60m)

        # Gap: positive = 60m more bullish than 5m
        gap = p60m - p5m

        # Optional 15m confirmation: 15m should be between 5m and 60m
        if require_15m and yes_15m is not None:
            p15m = float(yes_15m)
            # For a bullish 60m signal: need 15m also bullish (> 0.5 or > 5m)
            if gap > consistency_thresh and p15m < p5m:
                return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
            if gap < -consistency_thresh and p15m > p5m:
                return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)

        if gap > consistency_thresh:
            # 60m bullish but 5m hasn't moved → 5m is cheap → bet UP
            signal.iloc[-1] = 1
            size.iloc[-1] = size_val
        elif gap < -consistency_thresh:
            # 60m bearish but 5m hasn't moved → 5m is overpriced → bet DOWN
            signal.iloc[-1] = -1
            size.iloc[-1] = size_val

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
