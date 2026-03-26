"""Open Interest Rate-of-Change strategy.

Open interest (OI) tells us the total number of open derivative contracts.
The RATE OF CHANGE of OI, not the level, is the predictive signal:

OI spike + price flat:
  → New positions opening but price not moving → squeeze setup
  → Direction: follow funding rate (positive funding → longs crowded → DOWN;
    negative funding → shorts crowded → UP)

OI collapse + price flat:
  → Positions closing but price not reacting → unwinding
  → Direction: fade recent price direction (positions being closed out)

Requires candles enriched with oi, oi_roc, oi_zscore columns.
Use enrich_candles(..., include_oi=True, include_funding=True).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class OIRateOfChangeStrategy:
    """OI spike + price stall signals squeeze; OI collapse signals unwinding."""

    name = "oi_roc"
    description = "OI rate-of-change extremes combined with price-stall detection"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "oi_z_thresh": 2.0,          # OI z-score threshold
            "price_flat_thresh": 0.001,   # price move < 0.1% → "flat"
            "recent_dir_lookback": 3,     # bars for recent direction (unwinding signal)
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "oi_z_thresh": [1.5, 2.0, 2.5],
            "price_flat_thresh": [0.0005, 0.001, 0.002],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate OI rate-of-change signals.

        Args:
            candles: OHLCV DataFrame enriched with oi_zscore and ideally
                     funding_zscore columns. Use enrich_candles(..., include_oi=True).
            **params: oi_z_thresh, price_flat_thresh, recent_dir_lookback, size

        Returns:
            DataFrame with 'signal' (1/-1/0) and 'size' columns.
        """
        oi_z_thresh = float(params.get("oi_z_thresh", self.default_params["oi_z_thresh"]))
        price_flat_thresh = float(params.get("price_flat_thresh", self.default_params["price_flat_thresh"]))
        lookback = int(params.get("recent_dir_lookback", self.default_params["recent_dir_lookback"]))
        size_val = float(params.get("size", self.default_params["size"]))

        if "oi_zscore" not in candles.columns:
            return pd.DataFrame({"signal": 0, "size": 0.0}, index=candles.index)

        oi_z = candles["oi_zscore"]

        # Price change as absolute fraction
        price_chg = (candles["close"] / candles["close"].shift(1) - 1).abs()
        price_flat = price_chg < price_flat_thresh

        # Direction for squeeze: follow funding positioning
        if "funding_zscore" in candles.columns:
            # Positive funding → longs crowded → squeeze direction is DOWN
            # Negative funding → shorts crowded → squeeze direction is UP
            squeeze_dir = -np.sign(candles["funding_zscore"])
        else:
            # No funding data: use inverted recent price direction as proxy
            squeeze_dir = -np.sign(candles["close"] - candles["close"].shift(lookback))

        # Direction for unwinding: fade recent price move
        recent_dir = np.sign(candles["close"] - candles["close"].shift(lookback))

        signal = pd.Series(0, index=candles.index, dtype=int)

        oi_spike = oi_z > oi_z_thresh
        oi_collapse = oi_z < -oi_z_thresh

        # OI spike + price flat → squeeze incoming
        squeeze_mask = oi_spike & price_flat
        signal[squeeze_mask] = squeeze_dir[squeeze_mask].fillna(0).astype(int)

        # OI collapse + price flat → unwinding, fade direction
        unwind_mask = oi_collapse & price_flat
        signal[unwind_mask] = (-recent_dir[unwind_mask]).fillna(0).astype(int)

        signal = signal.fillna(0).astype(int)
        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
