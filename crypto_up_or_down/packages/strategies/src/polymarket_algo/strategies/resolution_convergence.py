"""Resolution Convergence strategy.

In the final seconds of a Polymarket 5m window, the YES/NO token price should
converge toward 0 or 1 as the outcome becomes clear from the Binance reference
price. When the price has clearly committed above or below the market's opening
price but YES/NO has not yet converged, a near-certain edge exists.

Entry conditions (all required):
  1. seconds_remaining <= trigger_seconds (in final window)
  2. Binance current price clearly above market open → outcome = UP
     OR Binance current price clearly below market open → outcome = DOWN
  3. Polymarket YES price has NOT converged to the implied outcome:
     For UP outcome: YES_price < yes_ceiling (e.g. 0.85)
     For DOWN outcome: YES_price > no_floor (e.g. 0.15)

This is a LIVE-ONLY strategy. The bot must inject runtime params:
  - yes_price: current Polymarket YES token mid-price (0–1)
  - seconds_remaining: seconds until this market's window closes
  - market_open_price: Binance reference price at the start of this window

Timing note:
  The bot fires at T-ENTRY_SECONDS_BEFORE (default 30s). Since trigger_seconds
  defaults to 45s, this strategy always gets at least one evaluation window.
"""

from __future__ import annotations

import pandas as pd


class ResolutionConvergenceStrategy:
    """Final-candle arbitrage: trade unconverged YES/NO prices near resolution."""

    name = "resolution_convergence"
    description = "Buy YES/NO when price has resolved but Polymarket hasn't converged yet"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "trigger_seconds": 45,    # activate in last N seconds of window
            "price_margin": 0.0001,   # min % move from open to consider "committed"
            "yes_ceiling": 0.85,      # UP outcome: YES is cheap if below this
            "no_floor": 0.15,         # DOWN outcome: YES is expensive if above this
            # Runtime params (injected by bot):
            "yes_price": None,         # float 0–1
            "seconds_remaining": None, # int
            "market_open_price": None, # float (Binance price at window open)
            "size": 20.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "trigger_seconds": [30, 45, 60],
            "yes_ceiling": [0.80, 0.85, 0.90],
            "no_floor": [0.10, 0.15, 0.20],
            "size": [15.0, 20.0, 25.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate resolution convergence opportunity.

        Args:
            candles: OHLCV DataFrame with current close price in last row.
            **params:
                trigger_seconds: activate threshold (seconds before close)
                price_margin: minimum fractional move from open to be "committed"
                yes_ceiling: max YES price to bet UP (i.e. YES cheap enough)
                no_floor: min YES price to bet DOWN (i.e. NO cheap enough)
                yes_price: current Polymarket YES mid (injected by bot)
                seconds_remaining: seconds until market closes (injected by bot)
                market_open_price: Binance price at window start (injected by bot)
                size: bet size in USD

        Returns:
            DataFrame with 'signal' (1/-1/0) and 'size' columns.
            Signal is only set on the last row; all other rows are 0.
        """
        trigger_seconds = int(params.get("trigger_seconds", self.default_params["trigger_seconds"]))
        price_margin = float(params.get("price_margin", self.default_params["price_margin"]))
        yes_ceiling = float(params.get("yes_ceiling", self.default_params["yes_ceiling"]))
        no_floor = float(params.get("no_floor", self.default_params["no_floor"]))
        yes_price = params.get("yes_price")
        seconds_remaining = params.get("seconds_remaining")
        market_open_price = params.get("market_open_price")
        size_val = float(params.get("size", self.default_params["size"]))

        signal = pd.Series(0, index=candles.index, dtype=int)
        size = pd.Series(0.0, index=candles.index)

        # Require all runtime params for live evaluation
        if yes_price is None or seconds_remaining is None or market_open_price is None:
            return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)

        # Only activate in final window
        if int(seconds_remaining) > trigger_seconds:
            return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)

        current_price = float(candles["close"].iloc[-1])
        ref_price = float(market_open_price)

        if ref_price <= 0:
            return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)

        price_move = (current_price - ref_price) / ref_price

        if price_move > price_margin:
            # Price clearly above open → likely UP outcome
            if float(yes_price) < yes_ceiling:
                signal.iloc[-1] = 1      # YES cheap → bet UP
                size.iloc[-1] = size_val
        elif price_move < -price_margin:
            # Price clearly below open → likely DOWN outcome
            if float(yes_price) > no_floor:
                signal.iloc[-1] = -1     # NO cheap → bet DOWN
                size.iloc[-1] = size_val

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
