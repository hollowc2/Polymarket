"""Candle-based composable gates for signal post-processing.

All gates follow the same interface as SessionFilter:
    gate.apply(signals: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame

They require only OHLCV columns (no enrichment needed) and are fully
backtest-compatible.

Classes:
    VolatilityGate  — veto signals when ATR is outside a normal range
    VolumeFilter    — veto signals on below-average volume candles
    TrendFilter     — veto signals that oppose the macro EMA trend
    VolAccelGate    — boost position size when realized vol suddenly accelerates
"""

from __future__ import annotations

import pandas as pd


def _atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    high = candles["high"]
    low = candles["low"]
    prev_close = candles["close"].shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


class VolatilityGate:
    """Veto signals outside a 'normal' ATR volatility band.

    Streaks in extremely low-volatility periods are often just drift and
    don't reverse cleanly. Streaks in extremely high-volatility periods
    have wide spreads and chaotic execution.

    The gate computes the rolling percentile rank of ATR over `window` bars
    and vetos candles below `low_pct` or above `high_pct`.

    Args:
        atr_period:  ATR calculation period (default 14)
        window:      Rolling lookback for percentile rank (default 288 = 1 day of 5m bars)
        low_pct:     Veto signals when ATR percentile < low_pct (too quiet)
        high_pct:    Veto signals when ATR percentile > high_pct (too chaotic)

    Usage:
        gate = VolatilityGate(low_pct=0.1, high_pct=0.9)
        signals = gate.apply(signals, candles)
    """

    def __init__(
        self,
        atr_period: int = 14,
        window: int = 288,
        low_pct: float = 0.1,
        high_pct: float = 0.9,
    ) -> None:
        self.atr_period = atr_period
        self.window = window
        self.low_pct = low_pct
        self.high_pct = high_pct

    def apply(self, signals: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        """Veto signals outside the normal ATR percentile band."""
        if not {"high", "low", "close"}.issubset(candles.columns):
            return signals

        atr = _atr(candles, self.atr_period)
        # Rolling percentile rank: fraction of past `window` ATR values below current ATR
        atr_pct = atr.rolling(self.window, min_periods=self.atr_period).rank(pct=True)

        too_quiet = atr_pct < self.low_pct
        too_chaotic = atr_pct > self.high_pct
        veto = too_quiet | too_chaotic

        out = signals.copy()
        out.loc[veto, "signal"] = 0
        if "size" in out.columns:
            out.loc[veto, "size"] = 0.0
        return out


class VolumeFilter:
    """Veto signals on below-average volume candles.

    A streak with thin volume has less conviction — the move may be noise
    from a single large order rather than genuine directional pressure.
    Requiring above-average volume improves signal quality at the cost of
    fewer trades.

    Args:
        window:    Rolling window for median volume (default 96 = 8h of 5m bars)
        min_mult:  Minimum volume relative to rolling median (default 1.0 = at median)
                   0.5 = above half-median, 1.5 = must be 50% above median, etc.

    Usage:
        gate = VolumeFilter(window=96, min_mult=1.0)
        signals = gate.apply(signals, candles)
    """

    def __init__(self, window: int = 96, min_mult: float = 1.0) -> None:
        self.window = window
        self.min_mult = min_mult

    def apply(self, signals: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        """Veto signals when candle volume is below rolling median × min_mult."""
        if "volume" not in candles.columns:
            return signals

        rolling_median = candles["volume"].rolling(self.window, min_periods=10).median()
        below_threshold = candles["volume"] < rolling_median * self.min_mult

        out = signals.copy()
        out.loc[below_threshold, "signal"] = 0
        if "size" in out.columns:
            out.loc[below_threshold, "size"] = 0.0
        return out


class TrendFilter:
    """Veto signals that oppose the macro EMA trend direction.

    Streak reversals are more reliable when they fade a counter-trend move
    than when they try to reverse the prevailing trend. This gate vetoes
    signals that would trade *with* a strong streak but *against* the
    macro direction.

    Mode "veto_counter_trend" (default):
        - If price > EMA (uptrend): veto signal = -1 (don't short into uptrend)
        - If price < EMA (downtrend): veto signal = +1 (don't buy into downtrend)
        Most conservative. Keeps only reversals that fight momentum AND align macro.

    Mode "veto_with_trend":
        - If price > EMA: veto signal = +1 (don't buy into an already-rising trend)
        - If price < EMA: veto signal = -1 (don't short into an already-falling trend)
        More aggressive. Only takes signals in the direction of the trend continuation.

    Args:
        ema_period: Period for macro trend EMA (default 200)
        mode:       "veto_counter_trend" or "veto_with_trend"

    Usage:
        gate = TrendFilter(ema_period=200, mode="veto_counter_trend")
        signals = gate.apply(signals, candles)
    """

    def __init__(
        self,
        ema_period: int = 200,
        mode: str = "veto_counter_trend",
    ) -> None:
        if mode not in ("veto_counter_trend", "veto_with_trend"):
            raise ValueError(f"Unknown mode: {mode!r}. Use 'veto_counter_trend' or 'veto_with_trend'.")
        self.ema_period = ema_period
        self.mode = mode

    def apply(self, signals: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        """Veto signals that oppose (or follow) the macro EMA trend."""
        if "close" not in candles.columns:
            return signals

        ema = candles["close"].ewm(span=self.ema_period, adjust=False).mean()

        # Live mode: signals use RangeIndex (synthetic candles from Polymarket outcomes),
        # real_candles use Timestamp index → pandas .loc[] aligns by label → no matches
        # → silent no-op. Fix: derive a single trend direction from the last candle and
        # broadcast it across all signal rows. Backtest: indices match, use row-by-row.
        if not candles.index.equals(signals.index):
            last_price_above = bool(candles["close"].iloc[-1] > ema.iloc[-1])
            price_above_ema = pd.Series(last_price_above, index=signals.index)
        else:
            price_above_ema = candles["close"] > ema

        out = signals.copy()

        if self.mode == "veto_counter_trend":
            # Uptrend → don't short (veto -1)
            out.loc[price_above_ema & (out["signal"] == -1), "signal"] = 0
            # Downtrend → don't buy (veto +1)
            out.loc[~price_above_ema & (out["signal"] == 1), "signal"] = 0
        else:  # "veto_with_trend"
            # Uptrend → don't buy the top (veto +1)
            out.loc[price_above_ema & (out["signal"] == 1), "signal"] = 0
            # Downtrend → don't short the bottom (veto -1)
            out.loc[~price_above_ema & (out["signal"] == -1), "signal"] = 0

        if "size" in out.columns:
            out.loc[out["signal"] == 0, "size"] = 0.0

        return out


class VolAccelGate:
    """Boost position size when realized vol suddenly accelerates.

    Computes vol_ratio = rolling_std(close, short_window) / rolling_std(close, long_window).
    When vol_ratio > threshold, multiplies `size` by boost_factor (capped at max_boost).
    Does NOT veto signals — only sizes up when Polymarket lag is highest.

    Note: When combining with VolatilityGate, use VolatilityGate(high_pct=0.99) or omit
    it entirely — VolatilityGate's high_pct veto conflicts with this gate's spike boost.

    Args:
        short_window: Bars for short-term vol (default 3 = 15 min at 5m)
        long_window:  Bars for baseline vol  (default 288 = 24h at 5m)
        threshold:    vol_ratio trigger       (default 2.0)
        boost_factor: Size multiplier         (default 1.5)
        max_boost:    Cap on total multiplier (default 3.0)
    """

    def __init__(
        self,
        short_window: int = 3,
        long_window: int = 288,
        threshold: float = 2.0,
        boost_factor: float = 1.5,
        max_boost: float = 3.0,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold
        self.boost_factor = boost_factor
        self.max_boost = max_boost

    def apply(self, signals: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        """Boost size on signals where vol has suddenly accelerated."""
        if "close" not in candles.columns:
            return signals

        returns = candles["close"].pct_change()
        short_vol = returns.rolling(self.short_window, min_periods=1).std()
        long_vol = returns.rolling(self.long_window, min_periods=self.long_window // 2).std()
        vol_ratio = short_vol / long_vol.clip(lower=1e-12)

        spike_mask = vol_ratio > self.threshold

        out = signals.copy()
        if "size" in out.columns:
            boosted = (out["size"] * self.boost_factor).clip(upper=out["size"] * self.max_boost)
            out.loc[spike_mask, "size"] = boosted.loc[spike_mask]

        return out
