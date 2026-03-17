"""Regression test: TrendFilter must veto signals when indices don't align (live mode)."""

import numpy as np
import pandas as pd
from polymarket_algo.strategies.gates import TrendFilter


def _make_candles(n: int, close_values: list[float]) -> pd.DataFrame:
    """Candles with Timestamp index (as returned by fetch_live_candles)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    closes = close_values + [close_values[-1]] * (n - len(close_values))
    return pd.DataFrame({"close": closes[:n]}, index=idx)


def _make_signals(n: int, signal_value: int) -> pd.DataFrame:
    """Signals with RangeIndex (as returned by outcomes_to_candles → strategy.evaluate)."""
    return pd.DataFrame(
        {"signal": [signal_value] * n, "size": [0.1] * n},
        index=range(n),
    )


# --- veto_with_trend (the live bot config) ---


def test_veto_with_trend_uptrend_kills_long_signals():
    """In uptrend (price > EMA), veto_with_trend should zero out signal=+1."""
    # Rising prices → last close well above EMA
    closes = list(np.linspace(90, 110, 60))  # 60 candles trending up
    candles = _make_candles(60, closes)
    signals = _make_signals(5, signal_value=1)

    gate = TrendFilter(ema_period=50, mode="veto_with_trend")
    result = gate.apply(signals, candles)

    assert (result["signal"] == 0).all(), "All +1 signals should be vetoed in uptrend"
    assert (result["size"] == 0.0).all()


def test_veto_with_trend_downtrend_kills_short_signals():
    """In downtrend (price < EMA), veto_with_trend should zero out signal=-1."""
    closes = list(np.linspace(110, 90, 60))  # trending down
    candles = _make_candles(60, closes)
    signals = _make_signals(5, signal_value=-1)

    gate = TrendFilter(ema_period=50, mode="veto_with_trend")
    result = gate.apply(signals, candles)

    assert (result["signal"] == 0).all(), "All -1 signals should be vetoed in downtrend"
    assert (result["size"] == 0.0).all()


def test_veto_with_trend_uptrend_passes_short_signals():
    """In uptrend, veto_with_trend should NOT veto signal=-1 (fade the trend)."""
    closes = list(np.linspace(90, 110, 60))
    candles = _make_candles(60, closes)
    signals = _make_signals(5, signal_value=-1)

    gate = TrendFilter(ema_period=50, mode="veto_with_trend")
    result = gate.apply(signals, candles)

    assert (result["signal"] == -1).all(), "-1 signals should pass through in uptrend"


# --- veto_counter_trend ---


def test_veto_counter_trend_uptrend_kills_short_signals():
    """In uptrend, veto_counter_trend should zero out signal=-1."""
    closes = list(np.linspace(90, 110, 60))
    candles = _make_candles(60, closes)
    signals = _make_signals(5, signal_value=-1)

    gate = TrendFilter(ema_period=50, mode="veto_counter_trend")
    result = gate.apply(signals, candles)

    assert (result["signal"] == 0).all()


# --- backtest mode (matching Timestamp indices) still works ---


def test_backtest_mode_matching_indices_unchanged():
    """When indices match, row-by-row logic should be used (backtest regression)."""
    idx = pd.date_range("2024-01-01", periods=60, freq="5min")
    closes = list(np.linspace(90, 110, 60))
    candles = pd.DataFrame({"close": closes}, index=idx)
    signals = pd.DataFrame({"signal": [1] * 60, "size": [0.1] * 60}, index=idx)

    gate = TrendFilter(ema_period=50, mode="veto_with_trend")
    result = gate.apply(signals, candles)

    # Most rows in an uptrend should be vetoed (price > EMA after warmup)
    assert (result["signal"] == 0).sum() > 0, "Gate should veto some signals in backtest mode"
