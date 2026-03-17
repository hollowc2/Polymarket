"""CVD (Cumulative Volume Delta) divergence strategy.

A bullish divergence occurs when price makes lower lows while CVD makes
higher lows (buyers are absorbing supply — reversal likely up).
A bearish divergence occurs when price makes higher highs while CVD makes
lower highs (sellers are overwhelming buyers despite rising price).

This strategy detects N-bar rolling divergence between price direction
and cumulative delta direction.

Requires candles enriched with delta and cvd columns.
Use enrich_candles() from packages/data to add these columns before evaluating.
"""

from __future__ import annotations

import pandas as pd


class CVDDivergenceStrategy:
    """Detect divergence between price momentum and order-flow delta.

    When price direction and delta direction disagree over `divergence_bars`
    consecutive candles, a reversal signal is generated.

    Signals:
        signal = -1  when price is rising but delta is falling (bearish divergence)
        signal = +1  when price is falling but delta is rising (bullish divergence)
        signal =  0  otherwise

    Requires candles["delta"] column (from enrich_candles()).
    """

    name = "cvd_divergence"
    description = "Reversal on CVD/price divergence — order flow disagrees with price direction"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "divergence_bars": 2,
            "delta_threshold": 0.0,
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "divergence_bars": [1, 2, 3],
            "delta_threshold": [0.0, 0.1, 0.5],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate CVD divergence strategy.

        Args:
            candles: OHLCV DataFrame enriched with delta column.
            **params: divergence_bars (int), delta_threshold (float), size (float)

        Returns:
            DataFrame with signal (int: 1/-1/0) and size (float) columns.
        """
        n_bars = int(params.get("divergence_bars", self.default_params["divergence_bars"]))
        threshold = float(params.get("delta_threshold", self.default_params["delta_threshold"]))
        size_val = float(params.get("size", self.default_params["size"]))

        if "delta" not in candles.columns:
            return pd.DataFrame(
                {"signal": 0, "size": 0.0},
                index=candles.index,
            )

        # Price direction: True if this close > previous close
        price_up = candles["close"].diff() > 0

        # Delta direction (vs threshold to filter noise)
        delta = candles["delta"]
        delta_pos = delta > threshold  # buying pressure
        delta_neg = delta < -threshold  # selling pressure

        # Rolling N-bar all(): all bars in window have same direction
        # pandas rolling().apply() with all → True when every element is 1
        def rolling_all(series: pd.Series, window: int) -> pd.Series:
            return (
                series.rolling(window, min_periods=window)
                .apply(lambda x: bool(x.all()), raw=True)
                .fillna(0)
                .astype(bool)
            )

        # Bearish divergence: price rising N bars, but delta consistently negative
        price_rising_n = rolling_all(price_up, n_bars)
        delta_falling_n = rolling_all(delta_neg, n_bars)
        bearish_div = price_rising_n & delta_falling_n

        # Bullish divergence: price falling N bars, but delta consistently positive
        price_falling_n = rolling_all(~price_up, n_bars)
        delta_rising_n = rolling_all(delta_pos, n_bars)
        bullish_div = price_falling_n & delta_rising_n

        signal = pd.Series(0, index=candles.index, dtype=int)
        signal[bearish_div] = -1
        signal[bullish_div] = 1

        # Bearish and bullish can't both be true (belt-and-suspenders)
        signal[bearish_div & bullish_div] = 0

        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
