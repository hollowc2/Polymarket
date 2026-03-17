"""Tests for microstructure data enrichers and enriched strategies.

These tests use synthetic data (no live API calls) to verify:
- Column presence after enrichment
- No NaNs at non-edge candles
- Signal output shape and dtype match the Strategy protocol
- Graceful degradation when enrichment columns are absent
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_candles(n: int = 100, freq: str = "5min") -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame suitable for all enrichers."""
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    price = 40_000 + rng.normal(0, 200, n).cumsum()
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.uniform(0, 100, n),
            "low": price - rng.uniform(0, 100, n),
            "close": price + rng.normal(0, 50, n),
            "volume": rng.uniform(1, 10, n),
        },
        index=idx,
    )
    return df


def make_funding_df(n: int = 10) -> pd.DataFrame:
    """Minimal funding rate DataFrame (8-hour snapshots)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"timestamp": idx, "funding_rate": rng.normal(0.0001, 0.0002, n)},
    )


def make_liq_df(n: int = 20) -> pd.DataFrame:
    """Minimal liquidations DataFrame."""
    rng = np.random.default_rng(2)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="13min", tz="UTC")
    sides = rng.choice(["BUY", "SELL"], n)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "side": sides,
            "price": rng.uniform(39_000, 41_000, n),
            "qty": rng.uniform(0.01, 2.0, n),
            "usd_value": rng.uniform(400, 80_000, n),
        }
    )


def make_trades_df(n: int = 200) -> pd.DataFrame:
    """Minimal aggTrades DataFrame."""
    rng = np.random.default_rng(3)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="90s", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": rng.uniform(39_000, 41_000, n),
            "qty": rng.uniform(0.001, 0.5, n),
            "is_buy_agg": rng.choice([True, False], n),
        }
    )


# --------------------------------------------------------------------------- #
# Data layer: compute_funding_candles
# --------------------------------------------------------------------------- #


class TestComputeFundingCandles:
    def test_adds_columns(self) -> None:
        from polymarket_algo.data.funding import compute_funding_candles

        candles = make_candles()
        funding = make_funding_df()
        out = compute_funding_candles(funding, candles)

        assert "funding_rate" in out.columns
        assert "funding_zscore" in out.columns

    def test_no_nans_mid_candles(self) -> None:
        from polymarket_algo.data.funding import compute_funding_candles

        candles = make_candles(200)
        funding = make_funding_df(30)
        out = compute_funding_candles(funding, candles)

        # Trim first/last 10 candles (edge effects from rolling window)
        mid = out.iloc[10:-10]
        assert not mid["funding_rate"].isna().any(), "NaN funding_rate in mid candles"

    def test_empty_funding_gives_zeros(self) -> None:
        from polymarket_algo.data.funding import compute_funding_candles

        candles = make_candles()
        empty_funding = pd.DataFrame(columns=["timestamp", "funding_rate"])
        out = compute_funding_candles(empty_funding, candles)

        assert (out["funding_rate"] == 0.0).all()
        assert (out["funding_zscore"] == 0.0).all()

    def test_index_preserved(self) -> None:
        from polymarket_algo.data.funding import compute_funding_candles

        candles = make_candles()
        funding = make_funding_df()
        out = compute_funding_candles(funding, candles)

        assert out.index.equals(candles.index)


# --------------------------------------------------------------------------- #
# Data layer: compute_liq_candles
# --------------------------------------------------------------------------- #


class TestComputeLiqCandles:
    def test_adds_columns(self) -> None:
        from polymarket_algo.data.liquidations import compute_liq_candles

        candles = make_candles()
        liqs = make_liq_df()
        out = compute_liq_candles(liqs, candles)

        assert "liq_long_usd" in out.columns
        assert "liq_short_usd" in out.columns
        assert "liq_net" in out.columns

    def test_no_negatives(self) -> None:
        from polymarket_algo.data.liquidations import compute_liq_candles

        candles = make_candles()
        liqs = make_liq_df()
        out = compute_liq_candles(liqs, candles)

        assert (out["liq_long_usd"] >= 0).all()
        assert (out["liq_short_usd"] >= 0).all()

    def test_net_equals_long_minus_short(self) -> None:
        from polymarket_algo.data.liquidations import compute_liq_candles

        candles = make_candles()
        liqs = make_liq_df()
        out = compute_liq_candles(liqs, candles)

        expected_net = out["liq_long_usd"] - out["liq_short_usd"]
        pd.testing.assert_series_equal(out["liq_net"], expected_net, check_names=False)

    def test_empty_liqs_gives_zeros(self) -> None:
        from polymarket_algo.data.liquidations import compute_liq_candles

        candles = make_candles()
        empty_liqs = pd.DataFrame(columns=["timestamp", "side", "price", "qty", "usd_value"])
        out = compute_liq_candles(empty_liqs, candles)

        assert (out["liq_long_usd"] == 0.0).all()
        assert (out["liq_short_usd"] == 0.0).all()

    def test_index_preserved(self) -> None:
        from polymarket_algo.data.liquidations import compute_liq_candles

        candles = make_candles()
        liqs = make_liq_df()
        out = compute_liq_candles(liqs, candles)

        assert out.index.equals(candles.index)


# --------------------------------------------------------------------------- #
# Data layer: compute_cvd_candles
# --------------------------------------------------------------------------- #


class TestComputeCVDCandles:
    def test_adds_columns(self) -> None:
        from polymarket_algo.data.agg_trades import compute_cvd_candles

        candles = make_candles()
        trades = make_trades_df()
        out = compute_cvd_candles(trades, candles)

        for col in ("buy_vol", "sell_vol", "delta", "cvd"):
            assert col in out.columns, f"Missing column: {col}"

    def test_delta_equals_buy_minus_sell(self) -> None:
        from polymarket_algo.data.agg_trades import compute_cvd_candles

        candles = make_candles()
        trades = make_trades_df()
        out = compute_cvd_candles(trades, candles)

        expected = out["buy_vol"] - out["sell_vol"]
        pd.testing.assert_series_equal(out["delta"], expected, check_names=False)

    def test_cvd_is_cumsum_of_delta(self) -> None:
        from polymarket_algo.data.agg_trades import compute_cvd_candles

        candles = make_candles()
        trades = make_trades_df()
        out = compute_cvd_candles(trades, candles)

        expected_cvd = out["delta"].cumsum()
        pd.testing.assert_series_equal(out["cvd"], expected_cvd, check_names=False)

    def test_empty_trades_gives_zeros(self) -> None:
        from polymarket_algo.data.agg_trades import compute_cvd_candles

        candles = make_candles()
        empty = pd.DataFrame(columns=["timestamp", "price", "qty", "is_buy_agg"])
        out = compute_cvd_candles(empty, candles)

        assert (out["delta"] == 0.0).all()
        assert (out["cvd"] == 0.0).all()


# --------------------------------------------------------------------------- #
# Strategy: FundingRateExtremesStrategy
# --------------------------------------------------------------------------- #


class TestFundingRateExtremesStrategy:
    def test_output_shape(self) -> None:
        from polymarket_algo.data.funding import compute_funding_candles
        from polymarket_algo.strategies.funding_rate_extremes import FundingRateExtremesStrategy

        strategy = FundingRateExtremesStrategy()
        candles = make_candles(100)
        funding = make_funding_df(15)
        enriched = compute_funding_candles(funding, candles)

        out = strategy.evaluate(enriched)
        assert set(out.columns) >= {"signal", "size"}
        assert len(out) == len(candles)

    def test_signals_are_valid_ints(self) -> None:
        from polymarket_algo.data.funding import compute_funding_candles
        from polymarket_algo.strategies.funding_rate_extremes import FundingRateExtremesStrategy

        strategy = FundingRateExtremesStrategy()
        candles = make_candles(100)
        funding = make_funding_df(15)
        enriched = compute_funding_candles(funding, candles)

        out = strategy.evaluate(enriched)
        assert out["signal"].isin([-1, 0, 1]).all()

    def test_size_zero_when_signal_zero(self) -> None:
        from polymarket_algo.data.funding import compute_funding_candles
        from polymarket_algo.strategies.funding_rate_extremes import FundingRateExtremesStrategy

        strategy = FundingRateExtremesStrategy()
        candles = make_candles(100)
        funding = make_funding_df(15)
        enriched = compute_funding_candles(funding, candles)

        out = strategy.evaluate(enriched)
        assert (out.loc[out["signal"] == 0, "size"] == 0.0).all()

    def test_graceful_degradation_missing_column(self) -> None:
        from polymarket_algo.strategies.funding_rate_extremes import FundingRateExtremesStrategy

        strategy = FundingRateExtremesStrategy()
        candles = make_candles(50)  # no funding_zscore column
        out = strategy.evaluate(candles)

        assert (out["signal"] == 0).all()

    def test_funding_filter_vetoes_long_signal_in_crowded_long(self) -> None:
        from polymarket_algo.strategies.funding_rate_extremes import FundingRateFilter

        idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
        candles = pd.DataFrame({"funding_zscore": [0.0, 0.0, 3.0, 3.0, 0.0]}, index=idx)
        signals = pd.DataFrame({"signal": [1, -1, 1, -1, 1], "size": [10.0] * 5}, index=idx)

        gate = FundingRateFilter(z_threshold=2.0)
        out = gate.apply(signals, candles)

        # Row 2 and 3 have zscore=3 (crowded longs):
        # signal=1 at row 2 should be vetoed → 0
        assert out.loc[idx[2], "signal"] == 0
        # signal=-1 at row 3 is not vetoed (direction matches fade)
        assert out.loc[idx[3], "signal"] == -1


# --------------------------------------------------------------------------- #
# Strategy: LiquidationCascadeStrategy
# --------------------------------------------------------------------------- #


class TestLiquidationCascadeStrategy:
    def test_output_shape(self) -> None:
        from polymarket_algo.data.liquidations import compute_liq_candles
        from polymarket_algo.strategies.liquidation_cascade import LiquidationCascadeStrategy

        strategy = LiquidationCascadeStrategy()
        candles = make_candles(100)
        liqs = make_liq_df(30)
        enriched = compute_liq_candles(liqs, candles)

        out = strategy.evaluate(enriched)
        assert set(out.columns) >= {"signal", "size"}
        assert len(out) == len(candles)

    def test_signals_are_valid_ints(self) -> None:
        from polymarket_algo.data.liquidations import compute_liq_candles
        from polymarket_algo.strategies.liquidation_cascade import LiquidationCascadeStrategy

        strategy = LiquidationCascadeStrategy()
        candles = make_candles(100)
        liqs = make_liq_df(30)
        enriched = compute_liq_candles(liqs, candles)

        out = strategy.evaluate(enriched)
        assert out["signal"].isin([-1, 0, 1]).all()

    def test_fade_mode_buy_after_long_liq(self) -> None:
        from polymarket_algo.strategies.liquidation_cascade import LiquidationCascadeStrategy

        idx = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
        candles = pd.DataFrame(
            {
                "close": [40000.0, 39000.0, 39500.0, 39800.0],
                "liq_long_usd": [0.0, 1_000_000.0, 0.0, 0.0],  # big long liq at row 1
                "liq_short_usd": [0.0, 0.0, 0.0, 0.0],
            },
            index=idx,
        )
        strategy = LiquidationCascadeStrategy()
        out = strategy.evaluate(candles, liq_threshold_usd=500_000, fade_cascade=True)

        # Signal should be +1 on candle AFTER the big liq (row 2)
        assert out.loc[idx[2], "signal"] == 1
        assert out.loc[idx[1], "signal"] == 0  # the liq candle itself has no signal

    def test_liquidation_gate_vetoes_during_cascade(self) -> None:
        from polymarket_algo.strategies.liquidation_cascade import LiquidationGate

        idx = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
        candles = pd.DataFrame(
            {
                "liq_long_usd": [0.0, 300_000.0, 0.0, 0.0],
                "liq_short_usd": [0.0, 0.0, 0.0, 0.0],
            },
            index=idx,
        )
        signals = pd.DataFrame({"signal": [1, 1, 1, 1], "size": [10.0] * 4}, index=idx)

        gate = LiquidationGate(cascade_usd=200_000)
        out = gate.apply(signals, candles)

        assert out.loc[idx[1], "signal"] == 0  # vetoed during cascade
        assert out.loc[idx[0], "signal"] == 1  # not affected


# --------------------------------------------------------------------------- #
# Strategy: CVDDivergenceStrategy
# --------------------------------------------------------------------------- #


class TestCVDDivergenceStrategy:
    def test_output_shape(self) -> None:
        from polymarket_algo.data.agg_trades import compute_cvd_candles
        from polymarket_algo.strategies.cvd_divergence import CVDDivergenceStrategy

        strategy = CVDDivergenceStrategy()
        candles = make_candles(100)
        trades = make_trades_df(500)
        enriched = compute_cvd_candles(trades, candles)

        out = strategy.evaluate(enriched)
        assert set(out.columns) >= {"signal", "size"}
        assert len(out) == len(candles)

    def test_signals_are_valid_ints(self) -> None:
        from polymarket_algo.data.agg_trades import compute_cvd_candles
        from polymarket_algo.strategies.cvd_divergence import CVDDivergenceStrategy

        strategy = CVDDivergenceStrategy()
        candles = make_candles(100)
        trades = make_trades_df(500)
        enriched = compute_cvd_candles(trades, candles)

        out = strategy.evaluate(enriched)
        assert out["signal"].isin([-1, 0, 1]).all()

    def test_graceful_degradation_missing_column(self) -> None:
        from polymarket_algo.strategies.cvd_divergence import CVDDivergenceStrategy

        strategy = CVDDivergenceStrategy()
        candles = make_candles(50)  # no delta column
        out = strategy.evaluate(candles)

        assert (out["signal"] == 0).all()

    def test_bearish_divergence_detected(self) -> None:
        """Manually construct a clear bearish divergence: price up 3 bars, delta negative."""
        from polymarket_algo.strategies.cvd_divergence import CVDDivergenceStrategy

        idx = pd.date_range("2024-01-01", periods=6, freq="5min", tz="UTC")
        # Bars 1-3: price rising, delta negative → bearish divergence
        candles = pd.DataFrame(
            {
                "close": [40000.0, 40100.0, 40200.0, 40300.0, 40200.0, 40100.0],
                "delta": [0.0, -1.0, -1.0, -1.0, 0.0, 0.0],
            },
            index=idx,
        )
        strategy = CVDDivergenceStrategy()
        out = strategy.evaluate(candles, divergence_bars=2, delta_threshold=0.0)

        # Bearish divergence should be detected at candle 2 or 3 (2-bar lookback)
        assert -1 in out["signal"].values


# --------------------------------------------------------------------------- #
# Strategy: CLOBImbalanceStrategy
# --------------------------------------------------------------------------- #


class TestCLOBImbalanceStrategy:
    def test_evaluate_returns_zero_signals(self) -> None:
        from polymarket_algo.strategies.clob_imbalance import CLOBImbalanceStrategy

        strategy = CLOBImbalanceStrategy()
        candles = make_candles(20)
        out = strategy.evaluate(candles)

        assert set(out.columns) >= {"signal", "size"}
        assert (out["signal"] == 0).all()

    def test_evaluate_book_abstains_with_none(self) -> None:
        from polymarket_algo.strategies.clob_imbalance import CLOBImbalanceStrategy

        strategy = CLOBImbalanceStrategy()
        assert strategy.evaluate_book(None) == 0

    def test_evaluate_book_signal_from_mock(self) -> None:
        """Test evaluate_book with a minimal mock orderbook."""
        import time

        from polymarket_algo.strategies.clob_imbalance import CLOBImbalanceStrategy

        class MockLevel:
            def __init__(self, price: float, size: float) -> None:
                self.price = price
                self.size = size

        class MockBook:
            best_bid = 0.49
            best_ask = 0.51
            mid = 0.50
            timestamp = time.time()
            # Deep bid side — strong YES imbalance
            bids = [MockLevel(0.49, 1000), MockLevel(0.48, 800)]
            asks = [MockLevel(0.51, 100), MockLevel(0.52, 50)]

        strategy = CLOBImbalanceStrategy()
        signal = strategy.evaluate_book(MockBook(), buy_threshold=1.5, depth_band=0.05)
        assert signal == 1  # yes bid_depth >> ask_depth → buy signal

    def test_spread_gate(self) -> None:

        import time

        from polymarket_algo.strategies.clob_imbalance import CLOBImbalanceStrategy

        class MockLevel:
            def __init__(self, price: float, size: float) -> None:
                self.price = price
                self.size = size

        class WideSpreadBook:
            best_bid = 0.40
            best_ask = 0.60
            mid = 0.50
            timestamp = time.time()
            bids = [MockLevel(0.40, 1000)]
            asks = [MockLevel(0.60, 100)]

        strategy = CLOBImbalanceStrategy()
        signal = strategy.evaluate_book(WideSpreadBook(), max_spread=0.03)
        assert signal == 0  # wide spread → abstain


# --------------------------------------------------------------------------- #
# Gate: VolAccelGate
# --------------------------------------------------------------------------- #


def _make_signals(n: int, signal_val: int = 1, size_val: float = 10.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"signal": [signal_val] * n, "size": [size_val] * n}, index=idx)


class TestVolAccelGate:
    def test_no_boost_when_quiet(self) -> None:
        """Flat-price candles → vol_ratio << threshold → size unchanged."""
        from polymarket_algo.strategies.gates import VolAccelGate

        n = 400
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        candles = pd.DataFrame({"close": [40_000.0] * n}, index=idx)
        signals = _make_signals(n, size_val=10.0)

        gate = VolAccelGate(threshold=2.0, boost_factor=1.5)
        out = gate.apply(signals, candles)

        # Flat price → pct_change = 0 → std = 0 → vol_ratio = 0, no boost
        assert (out["size"] == 10.0).all()

    def test_boost_applied_during_spike(self) -> None:
        """High-variance recent bars → vol_ratio > threshold → size boosted."""
        from polymarket_algo.strategies.gates import VolAccelGate

        rng = np.random.default_rng(99)
        n = 400
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        # Quiet base, then last 3 bars have extreme moves
        close = np.full(n, 40_000.0)
        close[-3:] += rng.normal(0, 2_000, 3)  # big spike
        candles = pd.DataFrame({"close": close}, index=idx)
        signals = _make_signals(n, size_val=10.0)

        gate = VolAccelGate(short_window=3, long_window=288, threshold=2.0, boost_factor=1.5)
        out = gate.apply(signals, candles)

        # At least one bar in the spike window should be boosted
        assert (out["size"] > 10.0).any()

    def test_boost_capped_at_max(self) -> None:
        """Extreme vol spike → boost is capped at max_boost × original size."""
        from polymarket_algo.strategies.gates import VolAccelGate

        rng = np.random.default_rng(7)
        n = 400
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        close = np.full(n, 40_000.0)
        close[-3:] += rng.normal(0, 50_000, 3)  # absurdly large spike
        candles = pd.DataFrame({"close": close}, index=idx)
        signals = _make_signals(n, size_val=10.0)

        gate = VolAccelGate(short_window=3, long_window=288, threshold=2.0, boost_factor=1.5, max_boost=3.0)
        out = gate.apply(signals, candles)

        assert (out["size"] <= 10.0 * 3.0).all()

    def test_missing_close_column(self) -> None:
        """No 'close' column → signals returned unchanged."""
        from polymarket_algo.strategies.gates import VolAccelGate

        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        candles = pd.DataFrame({"volume": [1.0] * n}, index=idx)
        signals = _make_signals(n, size_val=5.0)

        gate = VolAccelGate()
        out = gate.apply(signals, candles)

        pd.testing.assert_frame_equal(out, signals)

    def test_signal_zero_unchanged(self) -> None:
        """signal=0 rows remain zero-sized after boost."""
        from polymarket_algo.strategies.gates import VolAccelGate

        rng = np.random.default_rng(13)
        n = 400
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        close = np.full(n, 40_000.0)
        close[-3:] += rng.normal(0, 2_000, 3)
        candles = pd.DataFrame({"close": close}, index=idx)
        signals = pd.DataFrame({"signal": [0] * n, "size": [0.0] * n}, index=idx)

        gate = VolAccelGate(short_window=3, long_window=288, threshold=2.0, boost_factor=1.5)
        out = gate.apply(signals, candles)

        assert (out["size"] == 0.0).all()

    def test_index_preserved(self) -> None:
        """Output index matches input signals index exactly."""
        from polymarket_algo.strategies.gates import VolAccelGate

        n = 100
        candles = make_candles(n)
        signals = _make_signals(n)
        signals.index = candles.index

        gate = VolAccelGate()
        out = gate.apply(signals, candles)

        assert out.index.equals(signals.index)
