"""Realized Volatility Regime strategy.

Switches between momentum and mean-reversion based on the ratio of short-window
realized vol to long-window realized vol.

  ratio = RV(short) / RV(long)

  Low ratio  (vol compressing)   → breakout likely   → follow recent direction (momentum)
  High ratio (vol expanding)     → reversal likely   → fade recent direction (mean-reversion)
  Mid ratio                      → ambiguous regime  → no trade

Realized vol is computed as:
  RV(n) = sqrt( sum_{i=1}^{n} (log(close_i / close_{i-1}))^2 )

This is purely OHLCV-based — no new data sources required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RealizedVolRegimeStrategy:
    """Realized volatility regime: momentum when vol compresses, reversion when it expands."""

    name = "rv_regime"
    description = "Momentum in vol-compression regime; mean-reversion in vol-expansion regime"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "rv_short": 5,        # bars for short-window RV
            "rv_long": 60,        # bars for long-window RV
            "low_thresh": 0.7,    # RV ratio below this → momentum
            "high_thresh": 1.3,   # RV ratio above this → mean-reversion
            "lookback": 3,        # bars to determine recent direction
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "rv_short": [3, 5, 8],
            "rv_long": [40, 60, 80],
            "low_thresh": [0.6, 0.7, 0.8],
            "high_thresh": [1.2, 1.3, 1.5],
            "lookback": [2, 3, 5],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate realized vol regime signals.

        Args:
            candles: OHLCV DataFrame indexed by tz-aware UTC open_time.
                     Must contain 'close' column.
            **params: rv_short, rv_long, low_thresh, high_thresh, lookback, size

        Returns:
            DataFrame with 'signal' (1/-1/0) and 'size' columns.
        """
        rv_short = int(params.get("rv_short", self.default_params["rv_short"]))
        rv_long = int(params.get("rv_long", self.default_params["rv_long"]))
        low_thresh = float(params.get("low_thresh", self.default_params["low_thresh"]))
        high_thresh = float(params.get("high_thresh", self.default_params["high_thresh"]))
        lookback = int(params.get("lookback", self.default_params["lookback"]))
        size_val = float(params.get("size", self.default_params["size"]))

        log_ret = np.log(candles["close"] / candles["close"].shift(1))

        # Realized vol = sqrt(sum of squared log returns)
        rv_s = (log_ret**2).rolling(rv_short, min_periods=rv_short).sum() ** 0.5
        rv_l = (log_ret**2).rolling(rv_long, min_periods=rv_long // 2).sum() ** 0.5

        ratio = rv_s / rv_l.replace(0.0, float("nan"))

        # Recent direction: sign of (close - close[lookback bars ago])
        direction = np.sign(candles["close"] - candles["close"].shift(lookback))

        signal = pd.Series(0, index=candles.index, dtype=int)

        # Vol compressing → momentum: follow recent direction
        momentum_mask = ratio < low_thresh
        dir_int = direction.fillna(0).astype(int)
        signal[momentum_mask] = dir_int[momentum_mask]

        # Vol expanding → mean-reversion: fade recent direction
        reversion_mask = ratio > high_thresh
        signal[reversion_mask] = (-dir_int[reversion_mask])

        # Zero out where direction is flat (no signal)
        signal[dir_int == 0] = 0
        signal = signal.fillna(0).astype(int)

        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
