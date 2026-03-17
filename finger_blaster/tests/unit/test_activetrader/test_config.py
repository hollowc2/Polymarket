"""Comprehensive tests for AppConfig validation and defaults."""

import pytest
from src.activetrader.config import AppConfig


class TestAppConfigDefaults:
    """Test default configuration values."""

    def test_default_values_set(self):
        """Test all default values are set correctly."""
        config = AppConfig()

        # Trading defaults
        assert config.order_rate_limit_seconds == 0.5
        assert config.min_order_size == 1.0
        assert config.size_increment == 1.0

        # Analytics defaults
        assert config.analytics_interval == 0.5
        assert config.default_volatility == 0.60
        assert config.edge_threshold_bps == 50.0

        # Timer urgency
        assert config.timer_critical_minutes == 2
        assert config.timer_watchful_minutes == 5

        # WebSocket
        assert config.ws_ping_interval == 20
        assert config.ws_reconnect_delay == 5
        assert config.ws_max_reconnect_attempts == 10

        # History
        assert config.max_history_size == 10000

    def test_all_values_positive(self):
        """Test all numeric config values are positive."""
        config = AppConfig()

        assert config.order_rate_limit_seconds >= 0
        assert config.min_order_size > 0
        assert config.size_increment > 0
        assert config.analytics_interval > 0
        assert config.default_volatility > 0
        assert config.edge_threshold_bps >= 0
        assert config.timer_critical_minutes > 0
        assert config.timer_watchful_minutes > 0
        assert config.ws_ping_interval > 0
        assert config.ws_reconnect_delay >= 0
        assert config.ws_max_reconnect_attempts > 0
        assert config.max_history_size > 0


class TestAppConfigModification:
    """Test modifying configuration values."""

    def test_modify_rate_limit(self):
        """Test modifying rate limit."""
        config = AppConfig()
        config.order_rate_limit_seconds = 1.0

        assert config.order_rate_limit_seconds == 1.0

    def test_modify_analytics_interval(self):
        """Test modifying analytics interval."""
        config = AppConfig()
        config.analytics_interval = 1.0

        assert config.analytics_interval == 1.0

    def test_modify_volatility(self):
        """Test modifying default volatility."""
        config = AppConfig()
        config.default_volatility = 0.80

        assert config.default_volatility == 0.80

    def test_modify_edge_threshold(self):
        """Test modifying edge threshold."""
        config = AppConfig()
        config.edge_threshold_bps = 100.0

        assert config.edge_threshold_bps == 100.0

    def test_modify_timer_thresholds(self):
        """Test modifying timer urgency thresholds."""
        config = AppConfig()
        config.timer_critical_minutes = 1
        config.timer_watchful_minutes = 3

        assert config.timer_critical_minutes == 1
        assert config.timer_watchful_minutes == 3

    def test_modify_websocket_settings(self):
        """Test modifying WebSocket settings."""
        config = AppConfig()
        config.ws_ping_interval = 30
        config.ws_reconnect_delay = 10
        config.ws_max_reconnect_attempts = 5

        assert config.ws_ping_interval == 30
        assert config.ws_reconnect_delay == 10
        assert config.ws_max_reconnect_attempts == 5


class TestAppConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_rate_limit(self):
        """Test zero rate limit (no throttling)."""
        config = AppConfig()
        config.order_rate_limit_seconds = 0.0

        assert config.order_rate_limit_seconds == 0.0

    def test_very_high_volatility(self):
        """Test very high volatility setting."""
        config = AppConfig()
        config.default_volatility = 2.0  # 200% volatility

        assert config.default_volatility == 2.0

    def test_zero_edge_threshold(self):
        """Test zero edge threshold (always trade)."""
        config = AppConfig()
        config.edge_threshold_bps = 0.0

        assert config.edge_threshold_bps == 0.0

    def test_very_large_history_size(self):
        """Test very large history size."""
        config = AppConfig()
        config.max_history_size = 1000000

        assert config.max_history_size == 1000000

    def test_minimum_order_size_boundary(self):
        """Test minimum order size at boundary values."""
        config = AppConfig()

        # Set to very small value
        config.min_order_size = 0.01
        assert config.min_order_size == 0.01

        # Set to large value
        config.min_order_size = 1000.0
        assert config.min_order_size == 1000.0

    def test_fractional_analytics_interval(self):
        """Test fractional analytics interval."""
        config = AppConfig()
        config.analytics_interval = 0.1  # 100ms

        assert config.analytics_interval == 0.1


class TestAppConfigConsistency:
    """Test configuration consistency and relationships."""

    def test_timer_watchful_greater_than_critical(self):
        """Test that watchful threshold > critical threshold makes sense."""
        config = AppConfig()

        # Default should satisfy this
        assert config.timer_watchful_minutes > config.timer_critical_minutes

    def test_reconnect_attempts_reasonable(self):
        """Test reconnect attempts is reasonable."""
        config = AppConfig()

        # Should be positive and not excessive
        assert 1 <= config.ws_max_reconnect_attempts <= 100

    def test_ping_interval_less_than_timeout(self):
        """Test ping interval configuration."""
        config = AppConfig()

        # Ping interval should be reasonable
        assert config.ws_ping_interval > 0
        assert config.ws_ping_interval < 300  # Less than 5 minutes

    def test_history_size_reasonable(self):
        """Test history size is reasonable for 15min market."""
        config = AppConfig()

        # 15min market @ 1 update/sec = 900 updates
        # Default 10,000 should be sufficient
        assert config.max_history_size >= 1000  # At least 1000 points


class TestAppConfigValidation:
    """Test configuration validation (if implemented)."""

    def test_invalid_negative_values_accepted(self):
        """Test that negative values can be set (no validation enforced)."""
        config = AppConfig()

        # These may not make sense but are allowed
        config.min_order_size = -1.0
        assert config.min_order_size == -1.0

        # Note: In production, you might want to add validation
        # This test documents current behavior

    def test_volatility_range(self):
        """Test volatility can be set to any value."""
        config = AppConfig()

        # Very low
        config.default_volatility = 0.01
        assert config.default_volatility == 0.01

        # Very high
        config.default_volatility = 5.0
        assert config.default_volatility == 5.0

        # Negative (nonsensical but allowed)
        config.default_volatility = -0.5
        assert config.default_volatility == -0.5


class TestAppConfigTypes:
    """Test configuration value types."""

    def test_rate_limit_is_float(self):
        """Test rate limit accepts float."""
        config = AppConfig()
        config.order_rate_limit_seconds = 0.5

        assert isinstance(config.order_rate_limit_seconds, float)

    def test_min_order_size_is_float(self):
        """Test min order size accepts float."""
        config = AppConfig()
        config.min_order_size = 1.0

        assert isinstance(config.min_order_size, float)

    def test_analytics_interval_is_float(self):
        """Test analytics interval accepts float."""
        config = AppConfig()
        config.analytics_interval = 0.5

        assert isinstance(config.analytics_interval, float)

    def test_timer_minutes_are_int(self):
        """Test timer thresholds are integers."""
        config = AppConfig()

        assert isinstance(config.timer_critical_minutes, int)
        assert isinstance(config.timer_watchful_minutes, int)

    def test_websocket_settings_are_int(self):
        """Test WebSocket settings are integers."""
        config = AppConfig()

        assert isinstance(config.ws_ping_interval, int)
        assert isinstance(config.ws_reconnect_delay, int)
        assert isinstance(config.ws_max_reconnect_attempts, int)

    def test_history_size_is_int(self):
        """Test max history size is integer."""
        config = AppConfig()

        assert isinstance(config.max_history_size, int)


class TestAppConfigRealisticScenarios:
    """Test realistic configuration scenarios."""

    def test_high_frequency_trading_config(self):
        """Test config for high-frequency trading."""
        config = AppConfig()

        # Fast analytics
        config.analytics_interval = 0.1

        # No rate limit
        config.order_rate_limit_seconds = 0.0

        # Small min size
        config.min_order_size = 0.5

        assert config.analytics_interval == 0.1
        assert config.order_rate_limit_seconds == 0.0

    def test_conservative_trading_config(self):
        """Test config for conservative trading."""
        config = AppConfig()

        # Slow analytics
        config.analytics_interval = 2.0

        # High rate limit
        config.order_rate_limit_seconds = 5.0

        # High edge threshold
        config.edge_threshold_bps = 200.0

        assert config.analytics_interval == 2.0
        assert config.order_rate_limit_seconds == 5.0
        assert config.edge_threshold_bps == 200.0

    def test_high_volatility_market_config(self):
        """Test config for high volatility markets."""
        config = AppConfig()

        # High default volatility
        config.default_volatility = 1.2

        # Wider edge threshold
        config.edge_threshold_bps = 100.0

        assert config.default_volatility == 1.2
        assert config.edge_threshold_bps == 100.0

    def test_low_volatility_market_config(self):
        """Test config for low volatility markets."""
        config = AppConfig()

        # Low default volatility
        config.default_volatility = 0.3

        # Tighter edge threshold
        config.edge_threshold_bps = 25.0

        assert config.default_volatility == 0.3
        assert config.edge_threshold_bps == 25.0

    def test_unstable_network_config(self):
        """Test config for unstable network."""
        config = AppConfig()

        # Frequent pings
        config.ws_ping_interval = 10

        # More reconnect attempts
        config.ws_max_reconnect_attempts = 20

        # Faster reconnection
        config.ws_reconnect_delay = 2

        assert config.ws_ping_interval == 10
        assert config.ws_max_reconnect_attempts == 20
        assert config.ws_reconnect_delay == 2
