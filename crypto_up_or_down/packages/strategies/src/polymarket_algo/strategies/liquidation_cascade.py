"""Liquidation cascade strategy and composable gate.

Large liquidation cascades create two tradeable patterns:
1. Reversal (fade): After a cascade, price overshoots. Buy after long liqs,
   sell after short liqs.
2. Momentum (follow): Cascades can continue — ride the forced unwind direction.

The default is reversal (fade_cascade=True) which historically works better
in mean-reverting crypto markets after sharp deleveraging events.

Requires candles enriched with liq_long_usd, liq_short_usd, liq_net columns.
Use enrich_candles() from packages/data to add these columns before evaluating.

Classes:
    LiquidationCascadeStrategy — standalone strategy
    LiquidationGate            — composable filter that vetoes signals during
                                  active cascades (for use with other strategies)
"""

from __future__ import annotations

import pandas as pd

_LIQ_COLS = ("liq_long_usd", "liq_short_usd", "liq_net")


class LiquidationCascadeStrategy:
    """Standalone strategy trading around large liquidation cascade events.

    When a single candle sees a large USD liquidation volume, this strategy
    bets on a reversal (or continuation) in the following candle.

    Requires candles["liq_long_usd"] and candles["liq_short_usd"] columns
    (from enrich_candles()).

    Signals (fade_cascade=True, the default):
        signal = +1  after a large long-liquidation candle (buy the dip)
        signal = -1  after a large short-liquidation candle (sell the squeeze)
        signal =  0  otherwise
    """

    name = "liquidation_cascade"
    description = "Reversal/momentum trades around large liquidation cascade events"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "liq_threshold_usd": 500_000,
            "fade_cascade": True,
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "liq_threshold_usd": [200_000, 500_000, 1_000_000, 2_000_000],
            "fade_cascade": [True, False],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate liquidation cascade strategy.

        Args:
            candles: OHLCV DataFrame enriched with liq_long_usd and
                     liq_short_usd columns.
            **params: liq_threshold_usd, fade_cascade, size

        Returns:
            DataFrame with signal (int: 1/-1/0) and size (float) columns.
        """
        threshold = float(params.get("liq_threshold_usd", self.default_params["liq_threshold_usd"]))
        fade = bool(params.get("fade_cascade", self.default_params["fade_cascade"]))
        size_val = float(params.get("size", self.default_params["size"]))

        if not all(c in candles.columns for c in ("liq_long_usd", "liq_short_usd")):
            return pd.DataFrame(
                {"signal": 0, "size": 0.0},
                index=candles.index,
            )

        long_liqs = candles["liq_long_usd"]
        short_liqs = candles["liq_short_usd"]

        # Detect cascade candles (signal is generated on the NEXT candle)
        large_long_liq = long_liqs > threshold  # big long blowout
        large_short_liq = short_liqs > threshold  # big short blowout

        # Shift by 1: trade on the candle *after* the cascade
        signal = pd.Series(0, index=candles.index, dtype=int)

        if fade:
            # Fade: buy after longs get blown out (dip), sell after shorts get squeezed
            signal[large_long_liq.shift(1, fill_value=False)] = 1
            signal[large_short_liq.shift(1, fill_value=False)] = -1
        else:
            # Momentum: follow the cascade direction
            signal[large_long_liq.shift(1, fill_value=False)] = -1
            signal[large_short_liq.shift(1, fill_value=False)] = 1

        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)


class LiquidationGate:
    """Composable gate that vetoes signals during active liquidation cascades.

    During a cascade, spreads widen, slippage spikes, and price becomes
    chaotic. Vetoing other strategy signals during these windows reduces
    bad fills on normally-sound signals.

    Usage:
        signals = strategy.evaluate(candles)
        gate = LiquidationGate(cascade_usd=200_000)
        signals = gate.apply(signals, candles)

    Requires candles["liq_long_usd"] and candles["liq_short_usd"] columns.
    Falls back to pass-through if columns are missing.
    """

    def __init__(self, cascade_usd: float = 200_000) -> None:
        self.cascade_usd = cascade_usd

    def apply(self, signals: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        """Zero out signals on candles with active liquidation cascades."""
        if not all(c in candles.columns for c in ("liq_long_usd", "liq_short_usd")):
            return signals

        total_liq = candles["liq_long_usd"] + candles["liq_short_usd"]
        active_cascade = total_liq > self.cascade_usd

        out = signals.copy()
        out.loc[active_cascade, "signal"] = 0
        if "size" in out.columns:
            out.loc[active_cascade, "size"] = 0.0

        return out
