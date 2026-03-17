"""Comprehensive tests for Pulse GUI signal scoring."""

import pytest
from src.pulse.config import IndicatorSnapshot, Timeframe
from src.pulse.gui.scoring import (
    compute_signal_score,
    _compute_rsi_score,
    _compute_macd_score,
    _compute_trend_score,
    _compute_vwap_score,
    _get_label_and_description,
)


# ========== RSI Score Tests ==========
class TestComputeRSIScore:
    """Test RSI score calculation."""

    def test_rsi_zero_extreme_bullish(self):
        """Test RSI at 0 (extreme oversold) -> score 100."""
        score = _compute_rsi_score(0.0)
        assert score == 100.0

    def test_rsi_30_oversold_boundary(self):
        """Test RSI at 30 (oversold boundary) -> lower edge of neutral range."""
        score = _compute_rsi_score(30.0)
        # At RSI=30: 40 + (30-30) * (20/40) = 40
        assert score == 40.0

    def test_rsi_50_neutral(self):
        """Test RSI at 50 (neutral) -> score 50."""
        score = _compute_rsi_score(50.0)
        # At RSI=50: 40 + (50-30) * (20/40) = 40 + 20*0.5 = 50
        assert score == 50.0

    def test_rsi_70_overbought_boundary(self):
        """Test RSI at 70 (overbought boundary) -> upper edge of neutral range."""
        score = _compute_rsi_score(70.0)
        # At RSI=70: 40 + (70-30) * (20/40) = 40 + 40*0.5 = 60
        assert score == 60.0

    def test_rsi_100_extreme_bearish(self):
        """Test RSI at 100 (extreme overbought) -> score 0."""
        score = _compute_rsi_score(100.0)
        assert score == 0.0

    def test_rsi_15_highly_oversold(self):
        """Test RSI at 15 (highly oversold) -> bullish."""
        score = _compute_rsi_score(15.0)
        assert score > 70.0

    def test_rsi_85_highly_overbought(self):
        """Test RSI at 85 (highly overbought) -> bearish."""
        score = _compute_rsi_score(85.0)
        assert score < 30.0

    def test_rsi_in_neutral_zone(self):
        """Test RSI values in neutral zone (30-70)."""
        # RSI 40 should be slightly below neutral
        score_40 = _compute_rsi_score(40.0)
        assert 40.0 <= score_40 <= 50.0

        # RSI 60 should be slightly above neutral
        score_60 = _compute_rsi_score(60.0)
        assert 50.0 <= score_60 <= 60.0


# ========== MACD Score Tests ==========
class TestComputeMACDScore:
    """Test MACD histogram score calculation."""

    def test_macd_positive_bullish(self):
        """Test positive MACD histogram -> bullish score > 50."""
        score = _compute_macd_score(0.5)
        assert score > 50.0

    def test_macd_negative_bearish(self):
        """Test negative MACD histogram -> bearish score < 50."""
        score = _compute_macd_score(-0.5)
        assert score < 50.0

    def test_macd_zero_neutral(self):
        """Test zero MACD histogram -> neutral score 50."""
        score = _compute_macd_score(0.0)
        assert score == 50.0

    def test_macd_large_positive_capped_at_100(self):
        """Test large positive MACD is capped at 100."""
        score = _compute_macd_score(10.0)
        assert score == 100.0

    def test_macd_large_negative_capped_at_0(self):
        """Test large negative MACD is capped at 0."""
        score = _compute_macd_score(-10.0)
        assert score == 0.0

    def test_macd_small_positive(self):
        """Test small positive MACD histogram."""
        score = _compute_macd_score(0.1)
        assert 50.0 < score < 70.0

    def test_macd_small_negative(self):
        """Test small negative MACD histogram."""
        score = _compute_macd_score(-0.1)
        assert 30.0 < score < 50.0


# ========== Trend Score Tests ==========
class TestComputeTrendScore:
    """Test trend direction score calculation."""

    def test_trend_up_bullish(self):
        """Test UP trend -> bullish score."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            trend_direction="UP",
            adx=None,
        )
        score = _compute_trend_score(snapshot)
        assert score == 75.0

    def test_trend_down_bearish(self):
        """Test DOWN trend -> bearish score."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            trend_direction="DOWN",
            adx=None,
        )
        score = _compute_trend_score(snapshot)
        assert score == 25.0

    def test_trend_sideways_neutral(self):
        """Test SIDEWAYS trend -> neutral score."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            trend_direction="SIDEWAYS",
            adx=None,
        )
        score = _compute_trend_score(snapshot)
        assert score == 50.0

    def test_trend_up_amplified_by_high_adx(self):
        """Test UP trend amplified by high ADX."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            trend_direction="UP",
            adx=50.0,  # Strong trend
        )
        score = _compute_trend_score(snapshot)
        # Base 75, amplified by ADX factor (50/25 = 2.0)
        # 50 + (75 - 50) * 2.0 = 100
        assert score == 100.0

    def test_trend_down_amplified_by_high_adx(self):
        """Test DOWN trend amplified by high ADX."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            trend_direction="DOWN",
            adx=50.0,
        )
        score = _compute_trend_score(snapshot)
        # Base 25, amplified: 50 + (25 - 50) * 2.0 = 0
        assert score == 0.0

    def test_trend_with_low_adx_minimal_amplification(self):
        """Test trend with low ADX has minimal amplification."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            trend_direction="UP",
            adx=12.5,  # Half threshold
        )
        score = _compute_trend_score(snapshot)
        # Amplification factor: 12.5/25 = 0.5
        # 50 + (75 - 50) * 0.5 = 62.5
        assert abs(score - 62.5) < 0.01

    def test_trend_unknown_defaults_to_neutral(self):
        """Test unknown trend direction defaults to neutral."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            trend_direction="UNKNOWN",
            adx=None,
        )
        score = _compute_trend_score(snapshot)
        assert score == 50.0


# ========== VWAP Score Tests ==========
class TestComputeVWAPScore:
    """Test VWAP deviation score calculation."""

    def test_vwap_above_bullish(self):
        """Test price above VWAP -> bullish."""
        score = _compute_vwap_score(2.0)  # 2% above
        # 50 + 2.0 * 10 = 70
        assert score == 70.0

    def test_vwap_below_bearish(self):
        """Test price below VWAP -> bearish."""
        score = _compute_vwap_score(-2.0)  # 2% below
        # 50 + (-2.0) * 10 = 30
        assert score == 30.0

    def test_vwap_at_neutral(self):
        """Test price at VWAP -> neutral."""
        score = _compute_vwap_score(0.0)
        assert score == 50.0

    def test_vwap_strongly_above_capped_at_100(self):
        """Test strongly above VWAP capped at 100."""
        score = _compute_vwap_score(10.0)
        assert score == 100.0

    def test_vwap_strongly_below_capped_at_0(self):
        """Test strongly below VWAP capped at 0."""
        score = _compute_vwap_score(-10.0)
        assert score == 0.0

    def test_vwap_small_positive_deviation(self):
        """Test small positive deviation."""
        score = _compute_vwap_score(0.5)
        # 50 + 0.5 * 10 = 55
        assert score == 55.0

    def test_vwap_small_negative_deviation(self):
        """Test small negative deviation."""
        score = _compute_vwap_score(-0.5)
        # 50 + (-0.5) * 10 = 45
        assert score == 45.0


# ========== Label and Description Tests ==========
class TestGetLabelAndDescription:
    """Test label and description generation."""

    def test_strong_buy_label(self):
        """Test Strong Buy label for score >= 80."""
        label, description = _get_label_and_description(80)
        assert label == "Strong Buy"
        assert "bullish" in description.lower()

        label, description = _get_label_and_description(100)
        assert label == "Strong Buy"

    def test_bullish_label(self):
        """Test Bullish label for score 65-79."""
        label, _ = _get_label_and_description(65)
        assert label == "Bullish"

        label, _ = _get_label_and_description(79)
        assert label == "Bullish"

    def test_lean_bull_label(self):
        """Test Lean Bull label for score 55-64."""
        label, _ = _get_label_and_description(55)
        assert label == "Lean Bull"

        label, _ = _get_label_and_description(64)
        assert label == "Lean Bull"

    def test_neutral_label(self):
        """Test Neutral label for score 45-54."""
        label, _ = _get_label_and_description(45)
        assert label == "Neutral"

        label, _ = _get_label_and_description(54)
        assert label == "Neutral"

    def test_lean_bear_label(self):
        """Test Lean Bear label for score 35-44."""
        label, _ = _get_label_and_description(35)
        assert label == "Lean Bear"

        label, _ = _get_label_and_description(44)
        assert label == "Lean Bear"

    def test_bearish_label(self):
        """Test Bearish label for score 20-34."""
        label, _ = _get_label_and_description(20)
        assert label == "Bearish"

        label, _ = _get_label_and_description(34)
        assert label == "Bearish"

    def test_strong_sell_label(self):
        """Test Strong Sell label for score < 20."""
        label, description = _get_label_and_description(19)
        assert label == "Strong Sell"
        assert "bearish" in description.lower()

        label, _ = _get_label_and_description(0)
        assert label == "Strong Sell"


# ========== Compute Signal Score Integration Tests ==========
class TestComputeSignalScore:
    """Test complete signal score calculation."""

    def test_no_data_returns_neutral(self):
        """Test snapshot with minimal data returns neutral."""
        # Note: trend_direction defaults to "SIDEWAYS" which contributes to score
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
        )
        score, label, description = compute_signal_score(snapshot)
        assert score == 50
        # Only trend_direction (SIDEWAYS) contributes, resulting in neutral
        assert label == "Neutral"

    def test_bullish_signal_all_indicators(self):
        """Test bullish signal with all indicators aligned."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            rsi=25.0,  # Oversold (bullish)
            macd_histogram=0.5,  # Positive (bullish)
            trend_direction="UP",  # Bullish
            adx=30.0,  # Strong trend
            vwap_deviation=3.0,  # Above VWAP (bullish)
        )
        score, label, description = compute_signal_score(snapshot)
        assert score > 70
        assert label in ("Strong Buy", "Bullish")

    def test_bearish_signal_all_indicators(self):
        """Test bearish signal with all indicators aligned."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            rsi=80.0,  # Overbought (bearish)
            macd_histogram=-0.5,  # Negative (bearish)
            trend_direction="DOWN",  # Bearish
            adx=30.0,  # Strong trend
            vwap_deviation=-3.0,  # Below VWAP (bearish)
        )
        score, label, description = compute_signal_score(snapshot)
        assert score < 30
        assert label in ("Strong Sell", "Bearish")

    def test_neutral_signal_mixed_indicators(self):
        """Test neutral signal with mixed indicators."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            rsi=50.0,  # Neutral
            macd_histogram=0.0,  # Neutral
            trend_direction="SIDEWAYS",  # Neutral
            adx=15.0,  # Weak trend
            vwap_deviation=0.0,  # At VWAP
        )
        score, label, description = compute_signal_score(snapshot)
        assert 40 <= score <= 60
        assert label in ("Neutral", "Lean Bull", "Lean Bear")

    def test_partial_indicators(self):
        """Test with only some indicators available."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            rsi=30.0,  # Oversold boundary
            macd_histogram=0.2,  # Slightly positive
            # No trend or VWAP data
        )
        score, label, description = compute_signal_score(snapshot)
        assert 40 <= score <= 80

    def test_only_rsi_indicator(self):
        """Test with only RSI available."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            rsi=20.0,  # Oversold
        )
        score, label, description = compute_signal_score(snapshot)
        # Only RSI component, should be bullish from oversold
        assert score > 50

    def test_score_clamped_to_0_100(self):
        """Test score is clamped to 0-100 range."""
        # This tests extreme scenarios
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            rsi=0.0,
            macd_histogram=10.0,
            trend_direction="UP",
            adx=100.0,
            vwap_deviation=10.0,
        )
        score, _, _ = compute_signal_score(snapshot)
        assert 0 <= score <= 100

    def test_trend_direction_always_contributes(self):
        """Test trend_direction always has a value and contributes."""
        # Trend direction defaults to SIDEWAYS, so it always contributes
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            macd_histogram=0.0,  # Neutral
        )
        score, label, _ = compute_signal_score(snapshot)
        # Should have both MACD and trend components
        assert label != "No Data"


# ========== Edge Cases ==========
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_rsi_exactly_at_boundaries(self):
        """Test RSI at exact boundaries."""
        # RSI 0 (extreme oversold)
        assert _compute_rsi_score(0.0) == 100.0
        # RSI 30 (edge of neutral zone, maps to 40)
        assert _compute_rsi_score(30.0) == 40.0
        # RSI 70 (edge of neutral zone, maps to 60)
        assert _compute_rsi_score(70.0) == 60.0
        # RSI 100 (extreme overbought)
        assert _compute_rsi_score(100.0) == 0.0

    def test_macd_near_zero(self):
        """Test MACD values very close to zero."""
        assert _compute_macd_score(0.001) > 50.0
        assert _compute_macd_score(-0.001) < 50.0
        assert abs(_compute_macd_score(0.0) - 50.0) < 0.01

    def test_vwap_at_boundaries(self):
        """Test VWAP at calculation boundaries."""
        # +5% deviation maxes out
        assert _compute_vwap_score(5.0) == 100.0
        # -5% deviation bottoms out
        assert _compute_vwap_score(-5.0) == 0.0

    def test_adx_zero_no_amplification(self):
        """Test ADX of zero doesn't amplify trend (returns base score)."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            trend_direction="UP",
            adx=0.0,
        )
        # With ADX=0, condition `adx > 0` is False, so no amplification
        # Returns base score of 75 for UP trend
        score = _compute_trend_score(snapshot)
        assert score == 75.0

    def test_negative_adx_treated_as_no_amplification(self):
        """Test negative ADX is handled gracefully."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            trend_direction="UP",
            adx=-10.0,  # Invalid but shouldn't crash
        )
        # Should return base score without amplification
        score = _compute_trend_score(snapshot)
        # Negative ADX > 0 check fails, so no amplification
        assert score == 75.0

    def test_label_at_exact_boundaries(self):
        """Test labels at exact score boundaries."""
        # Test all boundary values
        boundaries = [
            (80, "Strong Buy"),
            (79, "Bullish"),
            (65, "Bullish"),
            (64, "Lean Bull"),
            (55, "Lean Bull"),
            (54, "Neutral"),
            (45, "Neutral"),
            (44, "Lean Bear"),
            (35, "Lean Bear"),
            (34, "Bearish"),
            (20, "Bearish"),
            (19, "Strong Sell"),
        ]
        for score, expected_label in boundaries:
            label, _ = _get_label_and_description(score)
            assert label == expected_label, f"Score {score}: expected {expected_label}, got {label}"
