"""Comprehensive tests for Pulse config data classes and enums."""

import pytest
import time
from src.pulse.config import (
    Timeframe,
    Candle,
    Trade,
    Ticker,
    BucketedOrderBook,
    IndicatorSnapshot,
    Alert,
    PulseConfig,
    PULSE_EVENTS,
)


# ========== Timeframe Enum Tests ==========
class TestTimeframe:
    """Test Timeframe enum properties."""

    def test_all_timeframes_have_seconds(self):
        """Test all timeframes return valid seconds."""
        expected = {
            Timeframe.TEN_SEC: 10,
            Timeframe.ONE_MIN: 60,
            Timeframe.FIVE_MIN: 300,
            Timeframe.FIFTEEN_MIN: 900,
            Timeframe.ONE_HOUR: 3600,
            Timeframe.FOUR_HOUR: 14400,
            Timeframe.ONE_DAY: 86400,
        }
        for tf, seconds in expected.items():
            assert tf.seconds == seconds

    def test_all_timeframes_have_coinbase_granularity(self):
        """Test all timeframes map to Coinbase granularity."""
        expected = {
            Timeframe.TEN_SEC: "ONE_MINUTE",
            Timeframe.ONE_MIN: "ONE_MINUTE",
            Timeframe.FIVE_MIN: "FIVE_MINUTE",
            Timeframe.FIFTEEN_MIN: "FIFTEEN_MINUTE",
            Timeframe.ONE_HOUR: "ONE_HOUR",
            Timeframe.FOUR_HOUR: "TWO_HOUR",
            Timeframe.ONE_DAY: "ONE_DAY",
        }
        for tf, granularity in expected.items():
            assert tf.coinbase_granularity == granularity

    def test_all_timeframes_have_display_name(self):
        """Test all timeframes have human-readable names."""
        expected = {
            Timeframe.TEN_SEC: "10 Second",
            Timeframe.ONE_MIN: "1 Minute",
            Timeframe.FIVE_MIN: "5 Minute",
            Timeframe.FIFTEEN_MIN: "15 Minute",
            Timeframe.ONE_HOUR: "1 Hour",
            Timeframe.FOUR_HOUR: "4 Hour",
            Timeframe.ONE_DAY: "Daily",
        }
        for tf, name in expected.items():
            assert tf.display_name == name

    def test_timeframe_values(self):
        """Test timeframe string values."""
        assert Timeframe.TEN_SEC.value == "10s"
        assert Timeframe.ONE_MIN.value == "1m"
        assert Timeframe.FIVE_MIN.value == "5m"
        assert Timeframe.FIFTEEN_MIN.value == "15m"
        assert Timeframe.ONE_HOUR.value == "1h"
        assert Timeframe.FOUR_HOUR.value == "4h"
        assert Timeframe.ONE_DAY.value == "1d"


# ========== Candle Dataclass Tests ==========
class TestCandle:
    """Test Candle dataclass."""

    @pytest.fixture
    def sample_candle(self):
        return Candle(
            timestamp=1700000000,
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50200.0,
            volume=100.0,
            timeframe=Timeframe.ONE_MIN,
        )

    def test_typical_price(self, sample_candle):
        """Test typical price calculation (HLC/3)."""
        expected = (50500.0 + 49500.0 + 50200.0) / 3
        assert sample_candle.typical_price == expected

    def test_hlc3_alias(self, sample_candle):
        """Test hlc3 is alias for typical_price."""
        assert sample_candle.hlc3 == sample_candle.typical_price

    def test_ohlc4(self, sample_candle):
        """Test OHLC4 calculation."""
        expected = (50000.0 + 50500.0 + 49500.0 + 50200.0) / 4
        assert sample_candle.ohlc4 == expected

    def test_to_dict(self, sample_candle):
        """Test dictionary conversion."""
        d = sample_candle.to_dict()
        assert d["timestamp"] == 1700000000
        assert d["open"] == 50000.0
        assert d["high"] == 50500.0
        assert d["low"] == 49500.0
        assert d["close"] == 50200.0
        assert d["volume"] == 100.0
        assert d["timeframe"] == "1m"

    def test_candle_is_frozen(self, sample_candle):
        """Test candle is immutable."""
        with pytest.raises(AttributeError):
            sample_candle.close = 51000.0

    def test_zero_volume_candle(self):
        """Test candle with zero volume."""
        candle = Candle(
            timestamp=1700000000,
            open=50000.0,
            high=50000.0,
            low=50000.0,
            close=50000.0,
            volume=0.0,
            timeframe=Timeframe.TEN_SEC,
        )
        assert candle.volume == 0.0
        assert candle.typical_price == 50000.0

    def test_doji_candle(self):
        """Test doji candle (open == close)."""
        candle = Candle(
            timestamp=1700000000,
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=10.0,
            timeframe=Timeframe.ONE_MIN,
        )
        assert candle.open == candle.close
        expected_tp = (50100.0 + 49900.0 + 50000.0) / 3
        assert candle.typical_price == expected_tp


# ========== Trade Dataclass Tests ==========
class TestTrade:
    """Test Trade dataclass."""

    @pytest.fixture
    def sample_trade(self):
        return Trade(
            trade_id="12345",
            product_id="BTC-USD",
            price=50000.0,
            size=0.5,
            side="BUY",
            timestamp=1700000000.123,
        )

    def test_trade_fields(self, sample_trade):
        """Test trade field access."""
        assert sample_trade.trade_id == "12345"
        assert sample_trade.product_id == "BTC-USD"
        assert sample_trade.price == 50000.0
        assert sample_trade.size == 0.5
        assert sample_trade.side == "BUY"
        assert sample_trade.timestamp == 1700000000.123

    def test_to_dict(self, sample_trade):
        """Test dictionary conversion."""
        d = sample_trade.to_dict()
        assert d["trade_id"] == "12345"
        assert d["product_id"] == "BTC-USD"
        assert d["price"] == 50000.0
        assert d["size"] == 0.5
        assert d["side"] == "BUY"
        assert d["timestamp"] == 1700000000.123

    def test_sell_trade(self):
        """Test sell trade."""
        trade = Trade(
            trade_id="67890",
            product_id="ETH-USD",
            price=3000.0,
            size=10.0,
            side="SELL",
            timestamp=1700000001.456,
        )
        assert trade.side == "SELL"


# ========== Ticker Dataclass Tests ==========
class TestTicker:
    """Test Ticker dataclass."""

    @pytest.fixture
    def sample_ticker(self):
        return Ticker(
            product_id="BTC-USD",
            price=50000.0,
            volume_24h=1000000.0,
            low_24h=49000.0,
            high_24h=51000.0,
            price_change_24h=500.0,
            price_change_pct_24h=1.01,
            timestamp=1700000000.0,
        )

    def test_ticker_fields(self, sample_ticker):
        """Test ticker field access."""
        assert sample_ticker.product_id == "BTC-USD"
        assert sample_ticker.price == 50000.0
        assert sample_ticker.volume_24h == 1000000.0
        assert sample_ticker.low_24h == 49000.0
        assert sample_ticker.high_24h == 51000.0
        assert sample_ticker.price_change_24h == 500.0
        assert sample_ticker.price_change_pct_24h == 1.01

    def test_to_dict(self, sample_ticker):
        """Test dictionary conversion."""
        d = sample_ticker.to_dict()
        assert d["product_id"] == "BTC-USD"
        assert d["price"] == 50000.0
        assert d["volume_24h"] == 1000000.0
        assert d["low_24h"] == 49000.0
        assert d["high_24h"] == 51000.0
        assert d["price_change_24h"] == 500.0
        assert d["price_change_pct_24h"] == 1.01

    def test_negative_price_change(self):
        """Test negative price change."""
        ticker = Ticker(
            product_id="BTC-USD",
            price=49000.0,
            volume_24h=500000.0,
            low_24h=48000.0,
            high_24h=50000.0,
            price_change_24h=-1000.0,
            price_change_pct_24h=-2.0,
            timestamp=1700000000.0,
        )
        assert ticker.price_change_24h == -1000.0
        assert ticker.price_change_pct_24h == -2.0


# ========== BucketedOrderBook Tests ==========
class TestBucketedOrderBook:
    """Test BucketedOrderBook dataclass."""

    @pytest.fixture
    def sample_book(self):
        return BucketedOrderBook(
            bids={50000.0: 10.0, 49900.0: 15.0, 49800.0: 20.0},
            asks={50100.0: 8.0, 50200.0: 12.0, 50300.0: 18.0},
            mid_price=50050.0,
            spread=100.0,
            best_bid=50000.0,
            best_ask=50100.0,
            timestamp=1700000000.0,
            bucket_size=100.0,
        )

    def test_spread_bps(self, sample_book):
        """Test spread in basis points."""
        # spread / mid_price * 10000
        expected_bps = (100.0 / 50050.0) * 10000
        assert abs(sample_book.spread_bps - expected_bps) < 0.01

    def test_spread_bps_zero_mid_price(self):
        """Test spread_bps with zero mid price."""
        book = BucketedOrderBook(
            bids={}, asks={},
            mid_price=0.0, spread=0.0,
            best_bid=0.0, best_ask=0.0,
            timestamp=1700000000.0, bucket_size=100.0,
        )
        assert book.spread_bps == 0.0

    def test_get_depth_at_level_bid(self, sample_book):
        """Test bid depth at price level."""
        # Within $100 of best bid (50000): min_price = 49900
        # Includes 50000.0 (10) and 49900.0 (15) since both >= 49900
        depth = sample_book.get_depth_at_level(100.0, side="bid")
        assert depth == 25.0  # 10 + 15

    def test_get_depth_at_level_bid_wider(self, sample_book):
        """Test bid depth with wider range."""
        # Within $200 of best bid (50000): min_price = 49800
        # Includes all three levels
        depth = sample_book.get_depth_at_level(200.0, side="bid")
        assert depth == 45.0  # 10 + 15 + 20

    def test_get_depth_at_level_ask(self, sample_book):
        """Test ask depth at price level."""
        # Within $100 of best ask (50100): max_price = 50200
        # Includes 50100.0 (8) and 50200.0 (12) since both <= 50200
        depth = sample_book.get_depth_at_level(100.0, side="ask")
        assert depth == 20.0  # 8 + 12

    def test_get_depth_at_level_ask_wider(self, sample_book):
        """Test ask depth with wider range."""
        # Within $200 of best ask (50100): max_price = 50300
        # Includes all three levels
        depth = sample_book.get_depth_at_level(200.0, side="ask")
        assert depth == 38.0  # 8 + 12 + 18

    def test_to_dict(self, sample_book):
        """Test dictionary conversion."""
        d = sample_book.to_dict()
        assert d["mid_price"] == 50050.0
        assert d["spread"] == 100.0
        assert d["best_bid"] == 50000.0
        assert d["best_ask"] == 50100.0
        assert d["bucket_size"] == 100.0
        assert "spread_bps" in d

    def test_empty_book(self):
        """Test empty order book."""
        book = BucketedOrderBook(
            bids={}, asks={},
            mid_price=0.0, spread=0.0,
            best_bid=0.0, best_ask=0.0,
            timestamp=1700000000.0, bucket_size=100.0,
        )
        assert book.get_depth_at_level(1000.0, "bid") == 0.0
        assert book.get_depth_at_level(1000.0, "ask") == 0.0


# ========== IndicatorSnapshot Tests ==========
class TestIndicatorSnapshot:
    """Test IndicatorSnapshot dataclass."""

    @pytest.fixture
    def sample_snapshot(self):
        return IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
            vwap=50000.0,
            vwap_deviation=0.5,
            adx=30.0,
            plus_di=25.0,
            minus_di=15.0,
            trend_direction="UP",
            atr=500.0,
            atr_pct=1.0,
            volatility=25.0,
            regime="TRENDING",
            rsi=65.0,
            macd_line=100.0,
            macd_signal=80.0,
            macd_histogram=20.0,
            bb_upper=51000.0,
            bb_middle=50000.0,
            bb_lower=49000.0,
        )

    def test_snapshot_fields(self, sample_snapshot):
        """Test snapshot field access."""
        assert sample_snapshot.product_id == "BTC-USD"
        assert sample_snapshot.timeframe == Timeframe.ONE_MIN
        assert sample_snapshot.vwap == 50000.0
        assert sample_snapshot.adx == 30.0
        assert sample_snapshot.trend_direction == "UP"
        assert sample_snapshot.regime == "TRENDING"
        assert sample_snapshot.rsi == 65.0

    def test_to_dict(self, sample_snapshot):
        """Test dictionary conversion."""
        d = sample_snapshot.to_dict()
        assert d["product_id"] == "BTC-USD"
        assert d["timeframe"] == "1m"
        assert d["vwap"] == 50000.0
        assert d["adx"] == 30.0
        assert d["rsi"] == 65.0
        assert d["macd_line"] == 100.0
        assert d["bb_upper"] == 51000.0

    def test_default_values(self):
        """Test snapshot with default values."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
        )
        assert snapshot.vwap is None
        assert snapshot.adx is None
        assert snapshot.trend_direction == "SIDEWAYS"
        assert snapshot.regime == "UNKNOWN"
        assert snapshot.rsi is None

    def test_snapshot_with_none_values(self):
        """Test snapshot serialization with None values."""
        snapshot = IndicatorSnapshot(
            product_id="BTC-USD",
            timeframe=Timeframe.ONE_MIN,
            timestamp=1700000000.0,
        )
        d = snapshot.to_dict()
        assert d["vwap"] is None
        assert d["rsi"] is None


# ========== Alert Tests ==========
class TestAlert:
    """Test Alert dataclass."""

    @pytest.fixture
    def sample_alert(self):
        return Alert(
            alert_type="regime_change",
            message="Market regime changed to TRENDING",
            product_id="BTC-USD",
            timestamp=1700000000.0,
            data={"previous_regime": "RANGING", "new_regime": "TRENDING"},
            severity="WARNING",
        )

    def test_alert_fields(self, sample_alert):
        """Test alert field access."""
        assert sample_alert.alert_type == "regime_change"
        assert sample_alert.message == "Market regime changed to TRENDING"
        assert sample_alert.product_id == "BTC-USD"
        assert sample_alert.severity == "WARNING"
        assert sample_alert.data["new_regime"] == "TRENDING"

    def test_to_dict(self, sample_alert):
        """Test dictionary conversion."""
        d = sample_alert.to_dict()
        assert d["alert_type"] == "regime_change"
        assert d["message"] == "Market regime changed to TRENDING"
        assert d["severity"] == "WARNING"
        assert d["data"]["new_regime"] == "TRENDING"

    def test_alert_default_severity(self):
        """Test alert with default severity."""
        alert = Alert(
            alert_type="volume_spike",
            message="Volume spike detected",
            product_id="ETH-USD",
            timestamp=1700000000.0,
        )
        assert alert.severity == "INFO"
        assert alert.data == {}

    def test_critical_alert(self):
        """Test critical severity alert."""
        alert = Alert(
            alert_type="level_breach",
            message="Price breached critical support",
            product_id="BTC-USD",
            timestamp=1700000000.0,
            severity="CRITICAL",
        )
        assert alert.severity == "CRITICAL"


# ========== PulseConfig Tests ==========
class TestPulseConfig:
    """Test PulseConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PulseConfig()
        assert config.products == ["BTC-USD"]
        assert Timeframe.ONE_MIN in config.enabled_timeframes
        assert config.candle_history_size == 200
        assert config.trade_history_size == 1000
        assert config.bucket_size_usd == 100.0
        assert config.rsi_period == 14
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9

    def test_custom_config(self):
        """Test custom configuration."""
        config = PulseConfig(
            products=["ETH-USD", "SOL-USD"],
            candle_history_size=500,
            rsi_period=21,
        )
        assert config.products == ["ETH-USD", "SOL-USD"]
        assert config.candle_history_size == 500
        assert config.rsi_period == 21

    def test_get_enabled_timeframes_list(self):
        """Test enabled timeframes sorted by seconds."""
        config = PulseConfig(
            enabled_timeframes={
                Timeframe.ONE_HOUR,
                Timeframe.ONE_MIN,
                Timeframe.FIVE_MIN,
            }
        )
        result = config.get_enabled_timeframes_list()
        assert result == [Timeframe.ONE_MIN, Timeframe.FIVE_MIN, Timeframe.ONE_HOUR]

    def test_all_timeframes_enabled(self):
        """Test config with all timeframes enabled."""
        config = PulseConfig(
            enabled_timeframes={
                Timeframe.TEN_SEC,
                Timeframe.ONE_MIN,
                Timeframe.FIVE_MIN,
                Timeframe.FIFTEEN_MIN,
                Timeframe.ONE_HOUR,
                Timeframe.FOUR_HOUR,
                Timeframe.ONE_DAY,
            }
        )
        result = config.get_enabled_timeframes_list()
        assert len(result) == 7
        assert result[0] == Timeframe.TEN_SEC  # Smallest first
        assert result[-1] == Timeframe.ONE_DAY  # Largest last


# ========== PULSE_EVENTS Tests ==========
class TestPulseEvents:
    """Test PULSE_EVENTS constant."""

    def test_events_is_tuple(self):
        """Test PULSE_EVENTS is a tuple."""
        assert isinstance(PULSE_EVENTS, tuple)

    def test_expected_events_exist(self):
        """Test expected events are defined."""
        expected = [
            'candle_update',
            'orderbook_update',
            'trade_update',
            'ticker_update',
            'indicator_update',
            'alert',
            'connection_status',
        ]
        for event in expected:
            assert event in PULSE_EVENTS

    def test_events_are_strings(self):
        """Test all events are strings."""
        for event in PULSE_EVENTS:
            assert isinstance(event, str)
