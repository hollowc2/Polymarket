"""Tests for ThreeBarMoMoStrategy."""

import pandas as pd
import pytest
from polymarket_algo.backtest.engine import run_backtest
from polymarket_algo.strategies.three_bar_momo import ThreeBarMoMoStrategy


def make_candles(
    opens: list[float],
    closes: list[float],
    volumes: list[float],
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from parallel lists."""
    assert len(opens) == len(closes) == len(volumes)
    idx = pd.date_range("2025-01-01", periods=len(opens), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )


@pytest.fixture
def strategy() -> ThreeBarMoMoStrategy:
    return ThreeBarMoMoStrategy()


# ---------------------------------------------------------------------------
# Direction tests
# ---------------------------------------------------------------------------


def test_bullish_setup_fires(strategy):
    """3 bullish bars with strictly increasing volume → signal=1 on last bar."""
    candles = make_candles(
        opens=[100.0, 101.0, 102.0],
        closes=[101.0, 102.0, 103.0],
        volumes=[10.0, 20.0, 30.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0)
    assert int(result.iloc[-1]["signal"]) == 1


def test_bearish_setup_fires(strategy):
    """3 bearish bars with strictly increasing volume → signal=-1 on last bar."""
    candles = make_candles(
        opens=[103.0, 102.0, 101.0],
        closes=[102.0, 101.0, 100.0],
        volumes=[10.0, 20.0, 30.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0)
    assert int(result.iloc[-1]["signal"]) == -1


# ---------------------------------------------------------------------------
# Volume condition
# ---------------------------------------------------------------------------


def test_flat_volume_no_signal(strategy):
    """Bullish bars but flat volume → no signal."""
    candles = make_candles(
        opens=[100.0, 101.0, 102.0],
        closes=[101.0, 102.0, 103.0],
        volumes=[10.0, 10.0, 10.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0)
    assert int(result.iloc[-1]["signal"]) == 0


def test_decreasing_volume_no_signal(strategy):
    """Bullish bars but decreasing volume → no signal."""
    candles = make_candles(
        opens=[100.0, 101.0, 102.0],
        closes=[101.0, 102.0, 103.0],
        volumes=[30.0, 20.0, 10.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0)
    assert int(result.iloc[-1]["signal"]) == 0


# ---------------------------------------------------------------------------
# Direction condition
# ---------------------------------------------------------------------------


def test_doji_middle_bar_no_signal(strategy):
    """Doji (open == close) in the middle → no signal."""
    candles = make_candles(
        opens=[100.0, 101.0, 101.0],
        closes=[101.0, 101.0, 102.0],  # middle bar is doji
        volumes=[10.0, 20.0, 30.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0)
    assert int(result.iloc[-1]["signal"]) == 0


def test_mixed_direction_no_signal(strategy):
    """Alternating direction → no signal."""
    candles = make_candles(
        opens=[100.0, 102.0, 101.0],
        closes=[102.0, 101.0, 103.0],  # up, down, up
        volumes=[10.0, 20.0, 30.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0)
    assert int(result.iloc[-1]["signal"]) == 0


# ---------------------------------------------------------------------------
# min_body_pct filter
# ---------------------------------------------------------------------------


def test_min_body_pct_blocks_tiny_candles(strategy):
    """Tiny body candles fail min_body_pct filter when enabled."""
    # Body is 0.01 / 100 = 0.0001 (0.01%), filter set to 0.001 (0.1%)
    candles = make_candles(
        opens=[100.00, 100.01, 100.02],
        closes=[100.01, 100.02, 100.03],
        volumes=[10.0, 20.0, 30.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0, min_body_pct=0.001)
    assert int(result.iloc[-1]["signal"]) == 0


def test_min_body_pct_passes_large_candles(strategy):
    """Large body candles pass the min_body_pct filter."""
    # Body is 1.0 / 100 = 0.01 (1%), filter at 0.001 (0.1%) → should pass
    candles = make_candles(
        opens=[100.0, 101.0, 102.0],
        closes=[101.0, 102.0, 103.0],
        volumes=[10.0, 20.0, 30.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0, min_body_pct=0.001)
    assert int(result.iloc[-1]["signal"]) == 1


def test_min_body_pct_zero_disabled(strategy):
    """min_body_pct=0.0 (default) does not filter out tiny candles."""
    candles = make_candles(
        opens=[100.00, 100.01, 100.02],
        closes=[100.01, 100.02, 100.03],
        volumes=[10.0, 20.0, 30.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0, min_body_pct=0.0)
    assert int(result.iloc[-1]["signal"]) == 1


# ---------------------------------------------------------------------------
# bars parameter
# ---------------------------------------------------------------------------


def test_bars_2_fires(strategy):
    """bars=2 with a 2-bar bullish+increasing-vol setup → signal=1."""
    candles = make_candles(
        opens=[100.0, 101.0],
        closes=[101.0, 102.0],
        volumes=[10.0, 20.0],
    )
    result = strategy.evaluate(candles, bars=2, size=15.0)
    assert int(result.iloc[-1]["signal"]) == 1


def test_bars_4_requires_four_bars(strategy):
    """bars=4: only 3 qualifying bars → no signal on last row."""
    candles = make_candles(
        opens=[100.0, 101.0, 102.0, 103.0],
        closes=[101.0, 102.0, 103.0, 104.0],
        volumes=[10.0, 20.0, 30.0, 25.0],  # 4th bar volume drops — no signal
    )
    result = strategy.evaluate(candles, bars=4, size=15.0)
    assert int(result.iloc[-1]["signal"]) == 0


def test_bars_4_fires_with_four_qualifying_bars(strategy):
    """bars=4: four qualifying bars → signal fires."""
    candles = make_candles(
        opens=[100.0, 101.0, 102.0, 103.0],
        closes=[101.0, 102.0, 103.0, 104.0],
        volumes=[10.0, 20.0, 30.0, 40.0],
    )
    result = strategy.evaluate(candles, bars=4, size=15.0)
    assert int(result.iloc[-1]["signal"]) == 1


# ---------------------------------------------------------------------------
# Rolling window: fires on every qualifying bar
# ---------------------------------------------------------------------------


def test_rolling_window_fires_on_continuation(strategy):
    """With 5 consecutive qualifying bars and bars=3, rows 2,3,4 all fire."""
    candles = make_candles(
        opens=[100.0, 101.0, 102.0, 103.0, 104.0],
        closes=[101.0, 102.0, 103.0, 104.0, 105.0],
        volumes=[10.0, 20.0, 30.0, 40.0, 50.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0)
    # rows 0,1 cannot fire (not enough history); rows 2,3,4 should all fire
    assert list(result["signal"].iloc[2:]) == [1, 1, 1]


# ---------------------------------------------------------------------------
# Size scaling
# ---------------------------------------------------------------------------


def test_size_scales_with_vol_ratio(strategy):
    """Size = base_size * (vol_last / vol_first), not exceeding size_cap."""
    candles = make_candles(
        opens=[100.0, 101.0, 102.0],
        closes=[101.0, 102.0, 103.0],
        volumes=[10.0, 20.0, 30.0],  # ratio = 30/10 = 3.0
    )
    result = strategy.evaluate(candles, bars=3, size=10.0, size_cap=2.0)
    # vol_ratio = 3.0, but capped at 2.0 → size = 10.0 * 2.0 = 20.0
    assert float(result.iloc[-1]["size"]) == pytest.approx(20.0)


def test_size_under_cap_uses_actual_ratio(strategy):
    """Size uses actual vol_ratio when it is below size_cap."""
    candles = make_candles(
        opens=[100.0, 101.0, 102.0],
        closes=[101.0, 102.0, 103.0],
        volumes=[10.0, 13.0, 15.0],  # strictly increasing; ratio = 15/10 = 1.5 < cap=2.0
    )
    result = strategy.evaluate(candles, bars=3, size=10.0, size_cap=2.0)
    assert float(result.iloc[-1]["size"]) == pytest.approx(15.0)


def test_no_signal_rows_have_zero_size(strategy):
    """Rows without a signal always have size=0."""
    candles = make_candles(
        opens=[100.0, 101.0, 100.0],  # last bar reverses
        closes=[101.0, 102.0, 99.0],
        volumes=[10.0, 20.0, 30.0],
    )
    result = strategy.evaluate(candles, bars=3, size=15.0)
    assert int(result.iloc[-1]["signal"]) == 0
    assert float(result.iloc[-1]["size"]) == 0.0


# ---------------------------------------------------------------------------
# Backtest smoke test
# ---------------------------------------------------------------------------


def test_backtest_smoke(strategy):
    """run_backtest with ThreeBarMoMoStrategy returns a valid BacktestResult."""
    n = 30
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    # alternating up/down with modest increasing volumes in blocks of 3
    opens = [100.0 + i * 0.5 for i in range(n)]
    closes = [o + (0.5 if i % 6 < 3 else -0.5) for i, o in enumerate(opens)]
    volumes = [10.0 + (i % 3) * 5.0 for i in range(n)]
    candles = pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )
    result = run_backtest(candles, strategy)
    assert "win_rate" in result.metrics
    assert "trade_count" in result.metrics
