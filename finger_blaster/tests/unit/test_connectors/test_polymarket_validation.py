"""Comprehensive tests for PolymarketConnector input validation.

Tests the @validate_order_params decorator, input validation across methods,
edge cases, and boundary conditions.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

from src.connectors.polymarket import (
    PolymarketConnector,
    validate_order_params,
    NetworkConstants,
    TradingConstants,
)


class TestValidateOrderParamsDecorator:
    """Test the @validate_order_params decorator."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.client.creds = MagicMock()
            conn.api = MagicMock()
            conn.session = MagicMock()
            conn.async_session = AsyncMock()
            conn._session_initialized = True
            conn.signature_type = 0
            return conn

    @pytest.mark.asyncio
    async def test_decorator_rejects_none_token_id(self, connector):
        """Test decorator rejects None token_id."""
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))

        result = await connector.create_market_order(None, 10.0, 'BUY')

        assert result is None

    @pytest.mark.asyncio
    async def test_decorator_rejects_empty_token_id(self, connector):
        """Test decorator rejects empty string token_id."""
        result = await connector.create_market_order('', 10.0, 'BUY')

        assert result is None

    @pytest.mark.asyncio
    async def test_decorator_rejects_short_token_id(self, connector):
        """Test decorator rejects token_id shorter than 10 chars."""
        result = await connector.create_market_order('0x12345', 10.0, 'BUY')

        assert result is None

    @pytest.mark.asyncio
    async def test_decorator_accepts_valid_token_id(self, connector):
        """Test decorator accepts valid token_id (>= 10 chars)."""
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))
        connector.client.create_market_order = MagicMock(return_value=MagicMock())
        connector.client.post_order = MagicMock(return_value={'orderID': 'test123'})

        # 64-char hex token (standard format)
        token_id = '0x' + '1' * 64
        result = await connector.create_market_order(token_id, 10.0, 'BUY')

        # Should proceed past decorator (may fail later, but decorator passed)
        assert result is not None or connector.get_order_book.called

    @pytest.mark.asyncio
    async def test_decorator_rejects_zero_amount(self, connector):
        """Test decorator rejects amount of 0."""
        result = await connector.create_market_order('0x' + '1' * 64, 0.0, 'BUY')

        assert result is None

    @pytest.mark.asyncio
    async def test_decorator_rejects_negative_amount(self, connector):
        """Test decorator rejects negative amount."""
        result = await connector.create_market_order('0x' + '1' * 64, -10.0, 'BUY')

        assert result is None

    @pytest.mark.asyncio
    async def test_decorator_rejects_excessive_amount(self, connector):
        """Test decorator rejects amount > 1,000,000."""
        result = await connector.create_market_order('0x' + '1' * 64, 1_000_001, 'BUY')

        assert result is None

    @pytest.mark.asyncio
    async def test_decorator_accepts_max_amount(self, connector):
        """Test decorator accepts exactly 1,000,000."""
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))
        connector.client.create_market_order = MagicMock(return_value=MagicMock())
        connector.client.post_order = MagicMock(return_value={'orderID': 'test123'})

        result = await connector.create_market_order('0x' + '1' * 64, 1_000_000, 'BUY')

        # Should pass decorator validation
        assert connector.get_order_book.called or result is not None

    @pytest.mark.asyncio
    async def test_decorator_accepts_min_amount(self, connector):
        """Test decorator accepts very small positive amount."""
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))
        connector.client.create_market_order = MagicMock(return_value=MagicMock())
        connector.client.post_order = MagicMock(return_value={'orderID': 'test123'})

        result = await connector.create_market_order('0x' + '1' * 64, 0.01, 'BUY')

        assert connector.get_order_book.called or result is not None

    @pytest.mark.asyncio
    async def test_decorator_rejects_invalid_side(self, connector):
        """Test decorator rejects invalid side."""
        result = await connector.create_market_order('0x' + '1' * 64, 10.0, 'HOLD')

        assert result is None

    @pytest.mark.asyncio
    async def test_decorator_accepts_buy_lowercase(self, connector):
        """Test decorator accepts 'buy' (lowercase)."""
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))
        connector.client.create_market_order = MagicMock(return_value=MagicMock())
        connector.client.post_order = MagicMock(return_value={'orderID': 'test123'})

        result = await connector.create_market_order('0x' + '1' * 64, 10.0, 'buy')

        assert connector.get_order_book.called or result is not None

    @pytest.mark.asyncio
    async def test_decorator_accepts_sell_uppercase(self, connector):
        """Test decorator accepts 'SELL' (uppercase)."""
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))
        connector.client.create_market_order = MagicMock(return_value=MagicMock())
        connector.client.post_order = MagicMock(return_value={'orderID': 'test123'})

        result = await connector.create_market_order('0x' + '1' * 64, 10.0, 'SELL')

        assert connector.get_order_book.called or result is not None

    @pytest.mark.asyncio
    async def test_decorator_rejects_empty_side(self, connector):
        """Test decorator rejects empty string side."""
        result = await connector.create_market_order('0x' + '1' * 64, 10.0, '')

        assert result is None


class TestCreateOrderValidation:
    """Test validation in create_order (limit order) method."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.client.creds = MagicMock()
            conn.api = MagicMock()
            return conn

    def test_create_order_rejects_empty_token_id(self, connector):
        """Test create_order rejects empty token_id."""
        result = connector.create_order('', 0.50, 10.0, 'BUY')

        assert result is None

    def test_create_order_rejects_none_token_id(self, connector):
        """Test create_order rejects None token_id."""
        result = connector.create_order(None, 0.50, 10.0, 'BUY')

        assert result is None

    def test_create_order_rejects_zero_price(self, connector):
        """Test create_order rejects price of 0."""
        result = connector.create_order('0x' + '1' * 64, 0.0, 10.0, 'BUY')

        assert result is None

    def test_create_order_rejects_negative_price(self, connector):
        """Test create_order rejects negative price."""
        result = connector.create_order('0x' + '1' * 64, -0.50, 10.0, 'BUY')

        assert result is None

    def test_create_order_rejects_zero_size(self, connector):
        """Test create_order rejects size of 0."""
        result = connector.create_order('0x' + '1' * 64, 0.50, 0.0, 'BUY')

        assert result is None

    def test_create_order_rejects_negative_size(self, connector):
        """Test create_order rejects negative size."""
        result = connector.create_order('0x' + '1' * 64, 0.50, -10.0, 'BUY')

        assert result is None

    def test_create_order_rejects_invalid_side(self, connector):
        """Test create_order rejects invalid side."""
        result = connector.create_order('0x' + '1' * 64, 0.50, 10.0, 'INVALID')

        assert result is None

    def test_create_order_accepts_valid_params(self, connector):
        """Test create_order accepts valid parameters."""
        connector.client.create_and_post_order = MagicMock(
            return_value={'orderID': 'test123'}
        )

        result = connector.create_order('0x' + '1' * 64, 0.50, 10.0, 'BUY')

        assert result == {'orderID': 'test123'}

    def test_create_order_accepts_sell_side(self, connector):
        """Test create_order accepts SELL side."""
        connector.client.create_and_post_order = MagicMock(
            return_value={'orderID': 'test123'}
        )

        result = connector.create_order('0x' + '1' * 64, 0.50, 10.0, 'SELL')

        assert result == {'orderID': 'test123'}


class TestCancelOrderValidation:
    """Test validation in cancel_order method."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            return conn

    def test_cancel_order_rejects_empty_order_id(self, connector):
        """Test cancel_order rejects empty order_id."""
        result = connector.cancel_order('')

        assert result is None

    def test_cancel_order_rejects_none_order_id(self, connector):
        """Test cancel_order rejects None order_id."""
        result = connector.cancel_order(None)

        assert result is None

    def test_cancel_order_accepts_valid_order_id(self, connector):
        """Test cancel_order accepts valid order_id."""
        connector.client.cancel = MagicMock(return_value={'canceled': True})

        result = connector.cancel_order('order_12345')

        assert result == {'canceled': True}


class TestGetOrderStatusValidation:
    """Test validation in get_order_status method."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            return conn

    def test_get_order_status_rejects_empty_order_id(self, connector):
        """Test get_order_status rejects empty order_id."""
        result = connector.get_order_status('')

        assert result is None

    def test_get_order_status_rejects_none_order_id(self, connector):
        """Test get_order_status rejects None order_id."""
        result = connector.get_order_status(None)

        assert result is None

    def test_get_order_status_accepts_valid_order_id(self, connector):
        """Test get_order_status accepts valid order_id."""
        connector.client.get_order = MagicMock(return_value={'status': 'filled'})

        result = connector.get_order_status('order_12345')

        assert result == {'status': 'filled'}


class TestTokenIdFormatValidation:
    """Test token ID format validation across methods."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.client.creds = MagicMock()
            conn.api = MagicMock()
            conn.async_session = AsyncMock()
            conn._session_initialized = True
            conn.signature_type = 0
            return conn

    @pytest.mark.asyncio
    async def test_valid_64char_hex_token(self, connector):
        """Test accepts valid 64-character hex token ID."""
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))
        connector.client.create_market_order = MagicMock(return_value=MagicMock())
        connector.client.post_order = MagicMock(return_value={'orderID': 'test'})

        token_id = '0x' + 'a' * 64
        result = await connector.create_market_order(token_id, 10.0, 'BUY')

        assert connector.get_order_book.called

    @pytest.mark.asyncio
    async def test_valid_mixed_case_hex_token(self, connector):
        """Test accepts mixed-case hex token ID."""
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))
        connector.client.create_market_order = MagicMock(return_value=MagicMock())
        connector.client.post_order = MagicMock(return_value={'orderID': 'test'})

        token_id = '0x' + 'AbCdEf1234' * 6 + 'AbCd'  # 64 chars after 0x
        result = await connector.create_market_order(token_id, 10.0, 'BUY')

        assert connector.get_order_book.called

    @pytest.mark.asyncio
    async def test_rejects_token_with_invalid_chars(self, connector):
        """Test rejects token ID with non-hex characters (but passes length check)."""
        # Note: Current implementation only checks length, not format
        # This test documents current behavior
        token_id = '0x' + 'g' * 64  # 'g' is not valid hex

        # Decorator passes (length check only), may fail later in execution
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))
        connector.client.create_market_order = MagicMock(return_value=MagicMock())
        connector.client.post_order = MagicMock(return_value={'orderID': 'test'})

        result = await connector.create_market_order(token_id, 10.0, 'BUY')

        # Documents that format validation is NOT currently done
        assert connector.get_order_book.called


class TestPriceBoundsValidation:
    """Test price boundary validation."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.client.creds = MagicMock()
            return conn

    def test_price_at_minimum_bound(self, connector):
        """Test accepts price at MIN_PRICE (0.01)."""
        connector.client.create_and_post_order = MagicMock(
            return_value={'orderID': 'test'}
        )

        result = connector.create_order(
            '0x' + '1' * 64, TradingConstants.MIN_PRICE, 10.0, 'BUY'
        )

        assert result is not None

    def test_price_at_maximum_bound(self, connector):
        """Test accepts price at MAX_PRICE (0.99)."""
        connector.client.create_and_post_order = MagicMock(
            return_value={'orderID': 'test'}
        )

        result = connector.create_order(
            '0x' + '1' * 64, TradingConstants.MAX_PRICE, 10.0, 'BUY'
        )

        assert result is not None

    def test_price_just_below_minimum(self, connector):
        """Test price just below MIN_PRICE is still accepted (no bounds check in create_order)."""
        # Documents that create_order does NOT enforce price bounds
        connector.client.create_and_post_order = MagicMock(
            return_value={'orderID': 'test'}
        )

        result = connector.create_order('0x' + '1' * 64, 0.001, 10.0, 'BUY')

        # Current implementation accepts this - gap identified
        assert result is not None

    def test_price_above_maximum(self, connector):
        """Test price above MAX_PRICE is still accepted (no bounds check in create_order)."""
        # Documents that create_order does NOT enforce price bounds
        connector.client.create_and_post_order = MagicMock(
            return_value={'orderID': 'test'}
        )

        result = connector.create_order('0x' + '1' * 64, 1.50, 10.0, 'BUY')

        # Current implementation accepts this - gap identified
        assert result is not None


class TestTimestampValidation:
    """Test timestamp validation in price fetch methods."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.gamma_url = NetworkConstants.GAMMA_API_URL
            conn.async_session = AsyncMock()
            conn._session_initialized = True
            conn.client = MagicMock()
            conn.api = MagicMock()
            return conn

    @pytest.mark.asyncio
    async def test_get_btc_price_at_past_timestamp(self, connector):
        """Test get_btc_price_at handles past timestamp."""
        # Mock _get_json_async to return klines data
        async def mock_get_json(*args, **kwargs):
            return [[1000, "50000.00", "50100.00", "49900.00", "50050.00", "100"]]

        connector._get_json_async = mock_get_json

        past_ts = pd.Timestamp('2025-01-01 12:00:00', tz='UTC')
        result = await connector.get_btc_price_at(past_ts)

        # str(float("50000.00")) = "50000.0"
        assert result == "50000.0"

    @pytest.mark.asyncio
    async def test_get_btc_price_at_future_timestamp(self, connector):
        """Test get_btc_price_at handles future timestamp (empty response)."""
        async def mock_get_json(*args, **kwargs):
            return []

        connector._get_json_async = mock_get_json

        future_ts = pd.Timestamp('2030-01-01 12:00:00', tz='UTC')
        result = await connector.get_btc_price_at(future_ts)

        assert result == "N/A"

    @pytest.mark.asyncio
    async def test_get_btc_price_at_timezone_aware(self, connector):
        """Test get_btc_price_at handles timezone-aware timestamp."""
        async def mock_get_json(*args, **kwargs):
            return [[1000, "50000.00", "50100.00", "49900.00", "50050.00", "100"]]

        connector._get_json_async = mock_get_json

        ts = pd.Timestamp('2025-01-01 12:00:00', tz='America/New_York')
        result = await connector.get_btc_price_at(ts)

        # str(float("50000.00")) = "50000.0"
        assert result == "50000.0"


class TestTokenMapValidation:
    """Test token map validation in flatten_market and related methods."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.api = MagicMock()
            conn.async_session = AsyncMock()
            conn._session_initialized = True
            conn.signature_type = 0
            return conn

    @pytest.mark.asyncio
    async def test_flatten_market_empty_token_map(self, connector):
        """Test flatten_market handles empty token map."""
        result = await connector.flatten_market({})

        assert result == []

    @pytest.mark.asyncio
    async def test_flatten_market_single_token(self, connector):
        """Test flatten_market handles token map with only one side."""
        connector.get_token_balance = AsyncMock(return_value=10.0)
        connector.create_market_order = AsyncMock(return_value={'orderID': 'test'})

        token_map = {'Up': '0x' + '1' * 64}
        result = await connector.flatten_market(token_map)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_flatten_market_zero_balances(self, connector):
        """Test flatten_market skips tokens with zero balance."""
        connector.get_token_balance = AsyncMock(return_value=0.0)

        token_map = {
            'Up': '0x' + '1' * 64,
            'Down': '0x' + '2' * 64
        }
        result = await connector.flatten_market(token_map)

        # No orders placed for zero balances
        assert result == []

    @pytest.mark.asyncio
    async def test_flatten_market_below_threshold(self, connector):
        """Test flatten_market skips balances below MIN_BALANCE_THRESHOLD."""
        connector.get_token_balance = AsyncMock(
            return_value=TradingConstants.MIN_BALANCE_THRESHOLD - 0.01
        )

        token_map = {'Up': '0x' + '1' * 64}
        result = await connector.flatten_market(token_map)

        assert result == []


class TestNullAndEmptyInputs:
    """Test handling of null and empty inputs across methods."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.api = MagicMock()
            conn.gamma_url = NetworkConstants.GAMMA_API_URL
            conn.async_session = AsyncMock()
            conn._session_initialized = True
            return conn

    def test_safe_parse_json_whitespace_only(self, connector):
        """Test _safe_parse_json_list handles whitespace-only string."""
        result = connector._safe_parse_json_list("   \t\n  ")

        assert result == []

    def test_safe_parse_json_list_nested_nulls(self, connector):
        """Test _safe_parse_json_list handles list with null values."""
        result = connector._safe_parse_json_list('[null, "value", null]')

        # Should convert items to strings, including 'None' for nulls
        assert len(result) == 3

    def test_build_token_map_none_tokens(self, connector):
        """Test _build_token_map handles None tokens array."""
        market = {'tokens': None}
        clob_ids = ['0x' + '1' * 64, '0x' + '2' * 64]

        result = connector._build_token_map(market, clob_ids)

        assert 'Up' in result and 'Down' in result

    def test_build_token_map_empty_clob_ids(self, connector):
        """Test _build_token_map handles empty clob_ids list."""
        market = {'tokens': []}
        clob_ids = []

        result = connector._build_token_map(market, clob_ids)

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_positions_empty_market_ids(self, connector):
        """Test get_positions handles empty market_ids list."""
        connector.api.wallet_address = '0x123'
        connector.api.get_positions = AsyncMock(return_value=[])

        result = await connector.get_positions(market_ids=[])

        assert result == []

    @pytest.mark.asyncio
    async def test_get_market_data_empty_id(self, connector):
        """Test get_market_data handles empty market_id."""
        async def mock_get_json(*args, **kwargs):
            return None

        connector._get_json_async = mock_get_json

        result = await connector.get_market_data("")

        assert result is None


class TestNumericEdgeCases:
    """Test numeric edge cases and floating point precision."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.client.creds = MagicMock()
            return conn

    def test_create_order_very_small_size(self, connector):
        """Test create_order accepts very small but positive size."""
        connector.client.create_and_post_order = MagicMock(
            return_value={'orderID': 'test'}
        )

        result = connector.create_order('0x' + '1' * 64, 0.50, 0.001, 'BUY')

        assert result is not None

    def test_create_order_large_size(self, connector):
        """Test create_order accepts large size."""
        connector.client.create_and_post_order = MagicMock(
            return_value={'orderID': 'test'}
        )

        result = connector.create_order('0x' + '1' * 64, 0.50, 999999.99, 'BUY')

        assert result is not None

    def test_create_order_precise_decimal(self, connector):
        """Test create_order handles precise decimal values."""
        connector.client.create_and_post_order = MagicMock(
            return_value={'orderID': 'test'}
        )

        result = connector.create_order('0x' + '1' * 64, 0.123456789, 10.987654321, 'BUY')

        assert result is not None

    def test_aggressive_price_very_low_bid(self, connector):
        """Test _calculate_aggressive_price with very low bid."""
        order_book = MagicMock()
        order_book.bids = [MagicMock(price='0.001')]

        price = connector._calculate_aggressive_price(order_book, 'SELL')

        # Should be floored at MIN_PRICE
        assert price == TradingConstants.MIN_PRICE

    def test_aggressive_price_very_high_ask(self, connector):
        """Test _calculate_aggressive_price with very high ask."""
        order_book = MagicMock()
        order_book.asks = [MagicMock(price='0.999')]

        price = connector._calculate_aggressive_price(order_book, 'BUY')

        # 0.999 * 1.10 = 1.0989, should be capped at MAX_PRICE
        assert price == TradingConstants.MAX_PRICE


class TestDecoratorEdgeCases:
    """Test edge cases in the validate_order_params decorator."""

    @pytest.mark.asyncio
    async def test_decorator_with_float_string_amount(self):
        """Test decorator behavior with edge case inputs."""

        class MockSelf:
            pass

        @validate_order_params
        async def test_func(self, token_id, amount, side):
            return "success"

        mock_self = MockSelf()

        # Valid inputs should pass
        result = await test_func(mock_self, '0x' + '1' * 64, 10.0, 'BUY')
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_with_exact_boundary_values(self):
        """Test decorator with boundary values."""

        class MockSelf:
            pass

        @validate_order_params
        async def test_func(self, token_id, amount, side):
            return "success"

        mock_self = MockSelf()

        # Exactly 10 characters (minimum)
        result = await test_func(mock_self, '0123456789', 1.0, 'BUY')
        assert result == "success"

        # 9 characters (below minimum)
        result = await test_func(mock_self, '012345678', 1.0, 'BUY')
        assert result is None

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""

        class MockSelf:
            pass

        @validate_order_params
        async def original_func(self, token_id, amount, side):
            """Original docstring."""
            return "success"

        # functools.wraps should preserve these
        assert original_func.__name__ == 'original_func'
        assert 'Original docstring' in original_func.__doc__


class TestExtractPriceToBeatValidation:
    """Test validation in _extract_price_to_beat method."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.api = MagicMock()
            return conn

    def test_extract_valid_numeric_threshold(self, connector):
        """Test extraction of valid numeric groupItemThreshold."""
        market = {'groupItemThreshold': 100000.50}

        result = connector._extract_price_to_beat(market, {})

        assert result == "100,000.50"

    def test_extract_valid_string_threshold(self, connector):
        """Test extraction of valid string groupItemThreshold."""
        market = {'groupItemThreshold': '$100,000.50'}

        result = connector._extract_price_to_beat(market, {})

        assert result == "100,000.50"

    def test_extract_zero_threshold(self, connector):
        """Test extraction handles zero threshold."""
        market = {'groupItemThreshold': 0}

        result = connector._extract_price_to_beat(market, {})

        # Zero should fall through to other strategies
        assert result == "N/A"

    def test_extract_negative_threshold(self, connector):
        """Test extraction handles negative threshold."""
        market = {'groupItemThreshold': -100}

        result = connector._extract_price_to_beat(market, {})

        # Negative should fall through to other strategies
        assert result == "N/A"

    def test_extract_from_title_pattern(self, connector):
        """Test extraction from title field."""
        market = {'title': 'BTC > $100,000'}

        result = connector._extract_price_to_beat(market, {})

        assert result == "100000"

    def test_extract_from_question_pattern(self, connector):
        """Test extraction from question field."""
        market = {'question': 'Will BTC be above $95,000?'}

        result = connector._extract_price_to_beat(market, {})

        assert result == "95000"
