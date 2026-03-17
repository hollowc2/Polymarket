"""Comprehensive tests for PolymarketConnector error handling.

Tests network errors, API failures, timeouts, and recovery scenarios.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import aiohttp
import pandas as pd

from src.connectors.polymarket import (
    PolymarketConnector,
    NetworkConstants,
    TradingConstants,
)


class TestNetworkErrors:
    """Test handling of network-level errors (timeout, connection refused, DNS)."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.gamma_url = NetworkConstants.GAMMA_API_URL
            conn.async_session = None
            conn._session_initialized = False
            conn.async_w3 = None
            conn.client = MagicMock()
            conn.api = MagicMock()
            conn.session = MagicMock()
            return conn

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp session."""
        session = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_get_btc_price_timeout(self, connector, mock_session):
        """Test BTC price fetch handles timeout gracefully."""
        connector.async_session = mock_session
        connector._session_initialized = True

        # Mock _get_json_async to return None (simulates timeout handled internally)
        async def mock_get_json(*args, **kwargs):
            return None

        connector._get_json_async = mock_get_json

        result = await connector.get_btc_price()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_get_btc_price_connection_error(self, connector, mock_session):
        """Test BTC price fetch handles connection refused."""
        connector.async_session = mock_session
        connector._session_initialized = True

        # Mock _get_json_async to return None (simulates error handled internally)
        async def mock_get_json(*args, **kwargs):
            return None

        connector._get_json_async = mock_get_json

        result = await connector.get_btc_price()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_get_active_market_network_error(self, connector, mock_session):
        """Test active market fetch handles network error."""
        connector.async_session = mock_session
        connector._session_initialized = True
        connector.api.get_active_market = AsyncMock(
            side_effect=aiohttp.ClientError("Network unreachable")
        )

        result = await connector.get_active_market()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_next_market_timeout(self, connector, mock_session):
        """Test next market fetch handles timeout."""
        # Create a proper async context manager mock
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock(side_effect=asyncio.TimeoutError())
        mock_response.json = AsyncMock(return_value=[])

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        mock_session.get.return_value = mock_context

        connector.async_session = mock_session
        connector._session_initialized = True
        # Prevent _ensure_async_session from creating a real session
        connector._ensure_async_session = AsyncMock()

        result = await connector.get_next_market("2026-01-25T12:00:00Z")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_closed_markets_connection_reset(self, connector, mock_session):
        """Test closed markets handles connection reset."""
        # Create a proper async context manager mock that raises on enter
        mock_context = AsyncMock()
        mock_context.__aenter__.side_effect = aiohttp.ServerDisconnectedError("Connection reset")
        mock_session.get.return_value = mock_context

        connector.async_session = mock_session
        connector._session_initialized = True
        # Prevent _ensure_async_session from creating a real session
        connector._ensure_async_session = AsyncMock()

        result = await connector.get_closed_markets()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_usdc_balance_web3_timeout(self, connector):
        """Test USDC balance handles Web3 timeout and falls back to CLOB."""
        connector.async_w3 = MagicMock()
        connector.async_w3.eth.contract.return_value.functions.balanceOf.return_value.call = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        connector.client.get_balance_allowance = MagicMock(return_value={'balance': 100.0})

        with patch.dict('os.environ', {'PRIVATE_KEY': '0x' + '1' * 64}):
            result = await connector.get_usdc_balance()

        # Should fallback to CLOB and return balance
        assert result == 100.0

    @pytest.mark.asyncio
    async def test_get_usdc_balance_both_fail(self, connector):
        """Test USDC balance returns 0 when both Web3 and CLOB fail."""
        connector.async_w3 = MagicMock()
        connector.async_w3.eth.contract.return_value.functions.balanceOf.return_value.call = AsyncMock(
            side_effect=Exception("Web3 failed")
        )
        connector.client.get_balance_allowance = MagicMock(
            side_effect=Exception("CLOB failed")
        )

        with patch.dict('os.environ', {'PRIVATE_KEY': '0x' + '1' * 64}):
            result = await connector.get_usdc_balance()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_flatten_market_partial_failure(self, connector):
        """Test flatten_market continues after individual token failure."""
        token_map = {
            'Up': '0x' + '1' * 64,
            'Down': '0x' + '2' * 64
        }

        # First token succeeds, second fails
        connector.get_token_balance = AsyncMock(side_effect=[10.0, 5.0])
        connector.create_market_order = AsyncMock(
            side_effect=[{'orderID': 'order1'}, None]
        )

        results = await connector.flatten_market(token_map)

        # Should have attempted both and returned results
        assert len(results) == 2
        assert results[0] == {'orderID': 'order1'}
        assert results[1] is None


class TestAPIErrors:
    """Test handling of API-level errors (429, 500, invalid responses)."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.gamma_url = NetworkConstants.GAMMA_API_URL
            conn.async_session = None
            conn._session_initialized = False
            conn.async_w3 = None
            conn.client = MagicMock()
            conn.api = MagicMock()
            conn.session = MagicMock()
            return conn

    @pytest.mark.asyncio
    async def test_get_btc_price_invalid_json(self, connector):
        """Test BTC price handles invalid JSON response."""
        async def mock_get_json(*args, **kwargs):
            return None  # Simulates failed JSON parse

        connector._get_json_async = mock_get_json
        connector.async_session = AsyncMock()
        connector._session_initialized = True

        result = await connector.get_btc_price()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_get_btc_price_missing_field(self, connector):
        """Test BTC price handles missing 'price' field."""
        async def mock_get_json(*args, **kwargs):
            return {'symbol': 'BTCUSDT'}  # Missing 'price' key

        connector._get_json_async = mock_get_json
        connector.async_session = AsyncMock()
        connector._session_initialized = True

        result = await connector.get_btc_price()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_get_btc_price_non_numeric_price(self, connector):
        """Test BTC price handles non-numeric price value."""
        async def mock_get_json(*args, **kwargs):
            return {'price': 'not_a_number'}

        connector._get_json_async = mock_get_json
        connector.async_session = AsyncMock()
        connector._session_initialized = True

        result = await connector.get_btc_price()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_get_active_market_empty_response(self, connector):
        """Test active market handles empty API response."""
        connector.async_session = AsyncMock()
        connector._session_initialized = True
        connector.api.get_active_market = AsyncMock(return_value=None)

        result = await connector.get_active_market()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_active_market_no_markets_array(self, connector):
        """Test active market handles response without 'markets' array."""
        connector.async_session = AsyncMock()
        connector._session_initialized = True
        connector.api.get_active_market = AsyncMock(return_value={'id': '123'})

        result = await connector.get_active_market()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_data_404(self, connector):
        """Test market data handles 404 response."""
        async def mock_get_json(*args, **kwargs):
            return None

        connector._get_json_async = mock_get_json
        connector.async_session = AsyncMock()
        connector._session_initialized = True

        result = await connector.get_market_data("nonexistent_market_id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_positions_invalid_wallet(self, connector):
        """Test positions handles missing wallet address."""
        connector.api.wallet_address = None
        connector.client.get_address = MagicMock(return_value=None)

        with patch.dict('os.environ', {}, clear=True):
            result = await connector.get_positions()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_btc_price_at_empty_klines(self, connector):
        """Test historical BTC price handles empty klines response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        connector.async_session = mock_session
        connector._session_initialized = True

        timestamp = pd.Timestamp.now(tz='UTC')
        result = await connector.get_btc_price_at(timestamp)

        assert result == "N/A"

    @pytest.mark.asyncio
    async def test_get_chainlink_price_at_all_approaches_fail(self, connector):
        """Test Chainlink price returns None when all approaches fail."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value={})

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        connector.async_session = mock_session
        connector._session_initialized = True

        timestamp = pd.Timestamp.now(tz='UTC')
        result = await connector.get_chainlink_price_at(timestamp)

        assert result is None


class TestSyncMethodErrors:
    """Test error handling in synchronous methods."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.api = MagicMock()
            conn.session = MagicMock()
            return conn

    def test_create_order_client_not_initialized(self, connector):
        """Test create_order handles uninitialized client."""
        connector.client = None

        result = connector.create_order('0x' + '1' * 64, 0.50, 10.0, 'BUY')

        assert result is None

    def test_create_order_missing_credentials(self, connector):
        """Test create_order handles missing API credentials."""
        connector.client.creds = None

        result = connector.create_order('0x' + '1' * 64, 0.50, 10.0, 'BUY')

        assert result is None

    def test_create_order_api_exception(self, connector):
        """Test create_order handles API exception."""
        connector.client.creds = MagicMock()
        connector.client.create_and_post_order = MagicMock(
            side_effect=Exception("API Error")
        )

        result = connector.create_order('0x' + '1' * 64, 0.50, 10.0, 'BUY')

        assert result is None

    def test_cancel_order_api_exception(self, connector):
        """Test cancel_order handles API exception."""
        connector.client.cancel = MagicMock(side_effect=Exception("Cancel failed"))

        result = connector.cancel_order("order_123")

        assert result is None

    def test_get_order_status_api_exception(self, connector):
        """Test get_order_status handles API exception."""
        connector.client.get_order = MagicMock(side_effect=Exception("Not found"))

        result = connector.get_order_status("order_123")

        assert result is None

    def test_get_order_book_api_exception(self, connector):
        """Test get_order_book handles API exception."""
        connector.client.get_order_book = MagicMock(
            side_effect=Exception("Order book unavailable")
        )

        result = connector.get_order_book('0x' + '1' * 64)

        assert result is None

    def test_get_latest_price_api_exception(self, connector):
        """Test get_latest_price handles API exception."""
        connector.client.get_midpoint = MagicMock(side_effect=Exception("Price unavailable"))

        result = connector.get_latest_price('0x' + '1' * 64)

        assert result == 0.0

    def test_fetch_data_api_exception(self, connector):
        """Test fetch_data handles API exception."""
        connector.client.get_trades = MagicMock(side_effect=Exception("Trades unavailable"))

        result = connector.fetch_data('0x' + '1' * 64, start_time=1000, end_time=2000)

        # Returns empty DataFrame on error
        assert result.empty

    def test_fetch_market_trades_events_exception(self, connector):
        """Test fetch_market_trades_events handles exception."""
        connector.client.get_market_trades_events = MagicMock(
            side_effect=Exception("Events unavailable")
        )

        result = connector.fetch_market_trades_events("condition_123")

        assert result == []


class TestAsyncMethodErrors:
    """Test error handling in async methods wrapped with to_thread."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.gamma_url = NetworkConstants.GAMMA_API_URL
            conn.async_session = None
            conn._session_initialized = False
            conn.async_w3 = None
            conn.client = MagicMock()
            conn.api = MagicMock()
            conn.session = MagicMock()
            conn.signature_type = 0
            return conn

    @pytest.mark.asyncio
    async def test_create_market_order_missing_credentials(self, connector):
        """Test create_market_order handles missing credentials."""
        connector.client.creds = None
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))

        # Decorator passes, but internal check catches missing creds
        result = await connector.create_market_order('0x' + '1' * 64, 10.0, 'BUY')

        assert result is None

    @pytest.mark.asyncio
    async def test_create_market_order_order_book_unavailable(self, connector):
        """Test create_market_order handles order book fetch failure."""
        connector.client.creds = MagicMock()
        connector.get_order_book = MagicMock(return_value=None)

        result = await connector.create_market_order('0x' + '1' * 64, 10.0, 'BUY')

        assert result is None

    @pytest.mark.asyncio
    async def test_create_market_order_signing_failure(self, connector):
        """Test create_market_order handles order signing failure."""
        connector.client.creds = MagicMock()
        connector.get_order_book = MagicMock(return_value=MagicMock(
            asks=[MagicMock(price='0.50')],
            bids=[MagicMock(price='0.49')]
        ))
        connector.client.create_market_order = MagicMock(
            side_effect=Exception("Signing failed")
        )

        result = await connector.create_market_order('0x' + '1' * 64, 10.0, 'BUY')

        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_all_orders_exception(self, connector):
        """Test cancel_all_orders handles exception."""
        connector.client.cancel_all = MagicMock(side_effect=Exception("Cancel all failed"))

        result = await connector.cancel_all_orders()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_token_balance_exception(self, connector):
        """Test get_token_balance handles exception."""
        connector.client.get_balance_allowance = MagicMock(
            side_effect=Exception("Balance check failed")
        )

        result = await connector.get_token_balance('0x' + '1' * 64)

        assert result == 0.0


class TestSessionManagement:
    """Test async session initialization and cleanup."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.gamma_url = NetworkConstants.GAMMA_API_URL
            conn.async_session = None
            conn._session_initialized = False
            conn.async_w3 = None
            conn.client = MagicMock()
            conn.api = MagicMock()
            conn.session = MagicMock()
            return conn

    @pytest.mark.asyncio
    async def test_ensure_async_session_creates_new(self, connector):
        """Test _ensure_async_session creates session when None."""
        mock_session = AsyncMock()
        connector._create_async_session = AsyncMock(return_value=mock_session)

        await connector._ensure_async_session()

        assert connector.async_session == mock_session
        assert connector._session_initialized is True

    @pytest.mark.asyncio
    async def test_ensure_async_session_reuses_existing(self, connector):
        """Test _ensure_async_session reuses non-closed session."""
        existing_session = MagicMock()
        # Ensure the session doesn't appear closed
        existing_session._closed = False
        # Also mock client_session for RetryClient check
        existing_session.client_session = MagicMock()
        existing_session.client_session._closed = False

        connector.async_session = existing_session
        connector._session_initialized = True
        connector._create_async_session = AsyncMock()

        await connector._ensure_async_session()

        # Should not create new session since existing one is not closed
        connector._create_async_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_handles_session_error(self, connector):
        """Test close handles errors during session cleanup."""
        mock_session = AsyncMock()
        mock_session.close = AsyncMock(side_effect=Exception("Close failed"))
        connector.async_session = mock_session

        # Should not raise
        await connector.close()

        assert connector.async_session is None

    @pytest.mark.asyncio
    async def test_close_handles_sync_session_error(self, connector):
        """Test close handles errors during sync session cleanup."""
        connector.async_session = None
        connector.session = MagicMock()
        connector.session.close = MagicMock(side_effect=Exception("Sync close failed"))

        # Should not raise
        await connector.close()


class TestParsingErrors:
    """Test error handling in parsing methods."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            conn.client = MagicMock()
            conn.api = MagicMock()
            return conn

    def test_safe_parse_json_list_invalid_json(self, connector):
        """Test _safe_parse_json_list handles invalid JSON."""
        result = connector._safe_parse_json_list("not valid json")

        assert result == []

    def test_safe_parse_json_list_empty_string(self, connector):
        """Test _safe_parse_json_list handles empty string."""
        result = connector._safe_parse_json_list("")

        assert result == []

    def test_safe_parse_json_list_none(self, connector):
        """Test _safe_parse_json_list handles None."""
        result = connector._safe_parse_json_list(None)

        assert result == []

    def test_safe_parse_json_list_not_string(self, connector):
        """Test _safe_parse_json_list handles non-string input."""
        result = connector._safe_parse_json_list(12345)

        assert result == []

    def test_safe_parse_json_list_non_list_json(self, connector):
        """Test _safe_parse_json_list handles JSON that's not a list."""
        result = connector._safe_parse_json_list('{"key": "value"}')

        assert result == []

    def test_parse_market_data_no_markets(self, connector):
        """Test _parse_market_data handles missing markets array."""
        result = connector._parse_market_data({'id': '123'})

        assert result is None

    def test_parse_market_data_empty_markets(self, connector):
        """Test _parse_market_data handles empty markets array."""
        result = connector._parse_market_data({'markets': []})

        assert result is None

    def test_parse_market_data_no_token_ids(self, connector):
        """Test _parse_market_data handles missing token IDs."""
        event = {
            'markets': [{'id': '123', 'clobTokenIds': '[]'}],
            'endDate': '2026-01-25T12:00:00Z'
        }

        result = connector._parse_market_data(event)

        assert result is None

    def test_build_token_map_empty_tokens(self, connector):
        """Test _build_token_map handles empty tokens array."""
        market = {'tokens': []}
        clob_ids = ['0x' + '1' * 64, '0x' + '2' * 64]

        result = connector._build_token_map(market, clob_ids)

        # Should fall back to clob_ids
        assert result == {'Up': clob_ids[0], 'Down': clob_ids[1]}

    def test_build_token_map_missing_outcome(self, connector):
        """Test _build_token_map handles tokens without outcome field."""
        market = {'tokens': [{'tokenId': '0x123'}]}
        clob_ids = ['0x' + '1' * 64]

        result = connector._build_token_map(market, clob_ids)

        # Should fall back to clob_ids for Up
        assert 'Up' in result

    def test_extract_price_to_beat_no_data(self, connector):
        """Test _extract_price_to_beat handles empty market data."""
        result = connector._extract_price_to_beat({}, {})

        assert result == "N/A"

    def test_extract_price_to_beat_invalid_threshold(self, connector):
        """Test _extract_price_to_beat handles invalid groupItemThreshold."""
        market = {'groupItemThreshold': 'not_a_number'}

        result = connector._extract_price_to_beat(market, {})

        # Should fall through to other strategies and return N/A
        assert result == "N/A"


class TestAggressivePriceCalculation:
    """Test edge cases in aggressive price calculation."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked dependencies."""
        with patch.object(PolymarketConnector, '__init__', lambda x: None):
            conn = PolymarketConnector()
            return conn

    def test_calculate_aggressive_price_empty_asks(self, connector):
        """Test aggressive BUY price with empty asks."""
        order_book = MagicMock()
        order_book.asks = []
        order_book.bids = [MagicMock(price='0.50')]

        price = connector._calculate_aggressive_price(order_book, 'BUY')

        # Should use MAX_PRICE when no asks
        assert price == TradingConstants.MAX_PRICE

    def test_calculate_aggressive_price_empty_bids(self, connector):
        """Test aggressive SELL price with empty bids."""
        order_book = MagicMock()
        order_book.asks = [MagicMock(price='0.50')]
        order_book.bids = []

        price = connector._calculate_aggressive_price(order_book, 'SELL')

        # Should use MIN_PRICE when no bids
        assert price == TradingConstants.MIN_PRICE

    def test_calculate_aggressive_price_buy_caps_at_max(self, connector):
        """Test aggressive BUY price is capped at MAX_PRICE."""
        order_book = MagicMock()
        order_book.asks = [MagicMock(price='0.95')]  # 0.95 * 1.10 = 1.045 > MAX

        price = connector._calculate_aggressive_price(order_book, 'BUY')

        assert price == TradingConstants.MAX_PRICE

    def test_calculate_aggressive_price_sell_floors_at_min(self, connector):
        """Test aggressive SELL price is floored at MIN_PRICE."""
        order_book = MagicMock()
        order_book.bids = [MagicMock(price='0.005')]  # 0.005 * 0.90 = 0.0045 < MIN

        price = connector._calculate_aggressive_price(order_book, 'SELL')

        assert price == TradingConstants.MIN_PRICE

    def test_calculate_aggressive_price_normal_buy(self, connector):
        """Test normal aggressive BUY price calculation."""
        order_book = MagicMock()
        order_book.asks = [MagicMock(price='0.50')]

        price = connector._calculate_aggressive_price(order_book, 'BUY')

        # 0.50 * 1.10 = 0.55
        assert price == 0.55

    def test_calculate_aggressive_price_normal_sell(self, connector):
        """Test normal aggressive SELL price calculation."""
        order_book = MagicMock()
        order_book.bids = [MagicMock(price='0.50')]

        price = connector._calculate_aggressive_price(order_book, 'SELL')

        # 0.50 * 0.90 = 0.45
        assert price == 0.45
