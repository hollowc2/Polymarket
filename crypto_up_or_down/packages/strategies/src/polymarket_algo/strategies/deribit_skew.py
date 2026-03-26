"""Deribit Options Skew strategy.

Options market participants pay more for protection in the direction they fear.
The 25-delta put/call risk reversal (skew) tells us which direction the options
market is pricing in:

  skew_25d = call_25d_IV - put_25d_IV

  skew < -5%: puts more expensive → bearish consensus → bet DOWN
  skew > +2%: calls more expensive → bullish consensus → bet UP

Additionally, the DVOL level (iv_atm) is used as a gate:
  - Very high IV (iv_zscore > gate_z) → too uncertain → no trade

Backtest behavior:
  The skew_25d column is 0.0 in backtests (no historical free-tier data).
  In backtest, strategy generates no trades — this is intentional. The strategy
  should be paper-traded live. iv_atm and iv_zscore are available historically
  and are used for the IV gate check.

Live behavior:
  The bot script injects the current live skew via params["live_skew"].
  The strategy uses this as an override when skew_25d column is all zeros.

Requires candles enriched with iv_atm, iv_zscore, skew_25d columns.
Use enrich_candles(..., include_deribit=True) before evaluating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DeribitSkewStrategy:
    """Options market 25-delta put/call skew as a directional signal."""

    name = "deribit_skew"
    description = "Trade Deribit put/call skew direction with DVOL regime gate"
    timeframe = "15m"

    @property
    def default_params(self) -> dict:
        return {
            "put_skew_thresh": -5.0,  # 25d RR below this → bearish
            "call_skew_thresh": 2.0,  # 25d RR above this → bullish
            "iv_gate_z": 2.0,         # skip when iv_zscore exceeds this
            "live_skew": None,        # injected by bot at runtime (float or None)
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "put_skew_thresh": [-8.0, -5.0, -3.0],
            "call_skew_thresh": [1.0, 2.0, 4.0],
            "iv_gate_z": [1.5, 2.0, 3.0],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate Deribit options skew signals.

        Args:
            candles: OHLCV DataFrame enriched with iv_atm, iv_zscore, skew_25d.
                     Use enrich_candles(..., include_deribit=True).
            **params:
                put_skew_thresh: bearish threshold for skew
                call_skew_thresh: bullish threshold for skew
                iv_gate_z: IV z-score gate (no trade if exceeded)
                live_skew: float — current live skew from Deribit (injected by bot)
                size: float

        Returns:
            DataFrame with 'signal' (1/-1/0) and 'size' columns.
        """
        put_skew_thresh = float(params.get("put_skew_thresh", self.default_params["put_skew_thresh"]))
        call_skew_thresh = float(params.get("call_skew_thresh", self.default_params["call_skew_thresh"]))
        iv_gate_z = float(params.get("iv_gate_z", self.default_params["iv_gate_z"]))
        live_skew = params.get("live_skew")
        size_val = float(params.get("size", self.default_params["size"]))

        signal = pd.Series(0, index=candles.index, dtype=int)
        size = pd.Series(0.0, index=candles.index)

        # IV gate: skip during very high or very low IV regimes
        if "iv_zscore" in candles.columns:
            iv_gate = candles["iv_zscore"].abs() > iv_gate_z
        else:
            iv_gate = pd.Series(False, index=candles.index)

        # Skew: prefer live_skew injected by bot; fall back to enriched column
        if live_skew is not None:
            # Live mode: use scalar skew for last-bar signal only
            if not iv_gate.iloc[-1]:
                current_skew = float(live_skew)
                if current_skew < put_skew_thresh:
                    signal.iloc[-1] = -1
                    size.iloc[-1] = size_val
                elif current_skew > call_skew_thresh:
                    signal.iloc[-1] = 1
                    size.iloc[-1] = size_val
        elif "skew_25d" in candles.columns:
            skew = candles["skew_25d"]
            # Backtest mode: skew column is 0.0 → no trades (intentional)
            has_skew_data = (skew.abs() > 0.01).any()
            if has_skew_data:
                signal[~iv_gate & (skew < put_skew_thresh)] = -1
                signal[~iv_gate & (skew > call_skew_thresh)] = 1
                signal = signal.fillna(0).astype(int)
                size[signal != 0] = size_val

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
