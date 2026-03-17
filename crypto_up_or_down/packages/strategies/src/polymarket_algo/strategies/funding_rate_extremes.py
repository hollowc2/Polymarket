"""Funding rate extremes strategy and composable gate.

Funding rates reflect market positioning bias. When funding is extremely
positive (crowded longs), the crowd is likely wrong and a fade makes sense.
When funding is extremely negative (crowded shorts), fade that too.

Requires candles enriched with funding_rate and funding_zscore columns.
Use enrich_candles() from packages/data to add these columns before evaluating.

Classes:
    FundingRateFilter  — composable gate (like SessionFilter), vetoes signals
                         that conflict with funding extremes.
    FundingRateExtremesStrategy — standalone contrarian strategy.
"""

from __future__ import annotations

import pandas as pd


class FundingRateFilter:
    """Composable gate that vetoes signals conflicting with funding extremes.

    When funding z-score > +threshold (crowded longs):
        - veto any signal=+1 (don't chase the crowd)
    When funding z-score < -threshold (crowded shorts):
        - veto any signal=-1 (don't chase the crowd)

    Usage:
        signals = strategy.evaluate(candles)
        frf = FundingRateFilter(z_threshold=2.0)
        signals = frf.apply(signals, candles)

    Requires candles["funding_zscore"] column (from enrich_candles()).
    Falls back to pass-through if column is missing.
    """

    def __init__(self, z_threshold: float = 2.0) -> None:
        self.z_threshold = z_threshold

    def apply(self, signals: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        """Veto signals that conflict with extreme funding positioning."""
        if "funding_zscore" not in candles.columns:
            return signals

        out = signals.copy()
        zscore = candles["funding_zscore"]

        # Crowded longs → veto UP bets
        crowded_long = zscore > self.z_threshold
        out.loc[crowded_long & (out["signal"] == 1), "signal"] = 0

        # Crowded shorts → veto DOWN bets
        crowded_short = zscore < -self.z_threshold
        out.loc[crowded_short & (out["signal"] == -1), "signal"] = 0

        # Zero size for vetoed signals
        if "size" in out.columns:
            out.loc[out["signal"] == 0, "size"] = 0.0

        return out


class FundingRateExtremesStrategy:
    """Standalone contrarian strategy: fade crowded funding positioning.

    Generates signals when funding z-score reaches extremes, betting
    against the crowded direction.

    Requires candles["funding_zscore"] column (from enrich_candles()).

    Signals:
        signal = -1  when zscore > +threshold (crowded longs, fade them)
        signal = +1  when zscore < -threshold (crowded shorts, fade them)
        signal =  0  otherwise
    """

    name = "funding_rate_extremes"
    description = "Contrarian fade at extreme funding rate z-score levels"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "z_threshold": 2.5,
            "size": 10.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "z_threshold": [1.5, 2.0, 2.5, 3.0],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate funding rate extremes strategy.

        Args:
            candles: OHLCV DataFrame (indexed by open_time) enriched with
                     funding_zscore column.
            **params: z_threshold (float), size (float)

        Returns:
            DataFrame with signal (int: 1/-1/0) and size (float) columns.
        """
        z_threshold = float(params.get("z_threshold", self.default_params["z_threshold"]))
        size_val = float(params.get("size", self.default_params["size"]))

        if "funding_zscore" not in candles.columns:
            # Graceful degradation: no data → no signals
            return pd.DataFrame(
                {"signal": 0, "size": 0.0},
                index=candles.index,
            )

        zscore = candles["funding_zscore"]
        signal = pd.Series(0, index=candles.index, dtype=int)

        # Crowded longs → expect reversal down
        signal[zscore > z_threshold] = -1
        # Crowded shorts → expect reversal up
        signal[zscore < -z_threshold] = 1

        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
