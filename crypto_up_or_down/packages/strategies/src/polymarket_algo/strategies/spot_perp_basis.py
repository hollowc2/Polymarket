"""Spot-Perp Basis strategy.

The basis = (perp_close - spot_close) / spot_close measures how much the
Binance perpetual futures contract trades at a premium or discount to spot.

When perps trade at a significant PREMIUM (positive basis z-score):
  → Longs are crowded and paying elevated funding costs
  → Contrarian: bet DOWN (fade the crowd)

When perps trade at a significant DISCOUNT (negative basis z-score):
  → Shorts are crowded and paying elevated funding to longs
  → Contrarian: bet UP (fade the crowd)

Requires candles enriched with basis and basis_zscore columns.
Use enrich_candles(..., include_basis=True) before evaluating.
"""

from __future__ import annotations

import pandas as pd


class SpotPerpBasisStrategy:
    """Contrarian fade when futures basis signals crowded positioning."""

    name = "spot_perp_basis"
    description = "Fade crowded longs/shorts when futures basis reaches extremes"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "z_thresh": 2.0,    # basis z-score threshold
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "z_thresh": [1.5, 2.0, 2.5, 3.0],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate spot-perp basis signals.

        Args:
            candles: OHLCV DataFrame enriched with basis_zscore column.
                     Use enrich_candles(..., include_basis=True).
            **params: z_thresh, size

        Returns:
            DataFrame with 'signal' (1/-1/0) and 'size' columns.
        """
        z_thresh = float(params.get("z_thresh", self.default_params["z_thresh"]))
        size_val = float(params.get("size", self.default_params["size"]))

        if "basis_zscore" not in candles.columns:
            return pd.DataFrame({"signal": 0, "size": 0.0}, index=candles.index)

        z = candles["basis_zscore"]
        signal = pd.Series(0, index=candles.index, dtype=int)

        # Futures at premium → longs crowded → fade DOWN
        signal[z > z_thresh] = -1
        # Futures at discount → shorts crowded → fade UP
        signal[z < -z_thresh] = 1

        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
