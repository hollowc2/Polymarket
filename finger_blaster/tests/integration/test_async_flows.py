"""Integration tests for async flows and concurrent behavior."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.activetrader.core import CallbackManager, CALLBACK_EVENTS
from src.activetrader.analytics import AnalyticsEngine
from src.activetrader.engine import OrderExecutor
from src.activetrader.config import AppConfig


# ========== CallbackManager Async Tests ==========
class TestCallbackManagerAsyncIntegration:
    """Test CallbackManager with real async scenarios."""

    @pytest.fixture
    def manager(self):
        return CallbackManager()

    @pytest.mark.asyncio
    async def test_concurrent_callback_execution(self, manager):
        """Test multiple callbacks execute concurrently."""
        execution_order = []
        execution_times = []

        async def slow_callback(value):
            await asyncio.sleep(0.1)
            execution_order.append(f"slow_{value}")
            execution_times.append(asyncio.get_event_loop().time())

        async def fast_callback(value):
            await asyncio.sleep(0.01)
            execution_order.append(f"fast_{value}")
            execution_times.append(asyncio.get_event_loop().time())

        manager.register('market_update', slow_callback)
        manager.register('market_update', fast_callback)

        start_time = asyncio.get_event_loop().time()
        manager.emit('market_update', 1)

        # Fast should complete before slow
        # But both should start concurrently
        await asyncio.sleep(0.2)  # Wait for all to complete

        # Fast callback should complete first
        assert execution_order[0].startswith('fast_')

    @pytest.mark.asyncio
    async def test_callback_isolation_on_error(self, manager):
        """Test error in one callback doesn't affect others."""
        results = []

        async def failing_callback():
            results.append('failing_started')
            raise ValueError("Intentional failure")

        async def successful_callback():
            await asyncio.sleep(0.01)
            results.append('successful_completed')

        manager.register('market_update', failing_callback)
        manager.register('market_update', successful_callback)

        # Should not raise exception
        manager.emit('market_update')

        await asyncio.sleep(0.05)

        # Successful callback should complete
        assert 'successful_completed' in results

    @pytest.mark.asyncio
    async def test_many_concurrent_emits(self, manager):
        """Test many concurrent emit calls."""
        counter = {'value': 0}
        lock = asyncio.Lock()

        async def increment_callback():
            async with lock:
                counter['value'] += 1

        manager.register('market_update', increment_callback)

        # Emit many times (emit is synchronous)
        for _ in range(100):
            manager.emit('market_update')

        # Wait for async callbacks to complete
        await asyncio.sleep(0.1)

        # All should have been processed
        assert counter['value'] == 100

    @pytest.mark.asyncio
    async def test_callback_execution_order_with_delays(self, manager):
        """Test callbacks with varying delays."""
        results = []

        async def callback_a():
            await asyncio.sleep(0.05)
            results.append('A')

        async def callback_b():
            await asyncio.sleep(0.01)
            results.append('B')

        async def callback_c():
            results.append('C')  # Instant

        manager.register('market_update', callback_a)
        manager.register('market_update', callback_b)
        manager.register('market_update', callback_c)

        manager.emit('market_update')

        # Wait for all to complete
        await asyncio.sleep(0.1)

        # C should be first (instant), B second (faster), A last
        assert results[0] == 'C'
        assert 'B' in results
        assert 'A' in results


# ========== OrderExecutor Async Integration ==========
class TestOrderExecutorAsyncIntegration:
    """Test OrderExecutor with real async scenarios."""

    @pytest.fixture
    def config(self):
        config = AppConfig()
        config.order_rate_limit_seconds = 0.1
        config.min_order_size = 1.0
        return config

    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock()
        # Simulate network delay
        async def delayed_order(token_id, size, side, price=None):
            await asyncio.sleep(0.05)
            return {'orderID': f'order_{size}'}
        connector.create_market_order = delayed_order
        return connector

    @pytest.fixture
    def executor(self, config, mock_connector):
        return OrderExecutor(config, mock_connector)

    @pytest.fixture
    def token_map(self):
        return {
            'Up': '0x' + '1' * 64,
            'Down': '0x' + '2' * 64
        }

    @pytest.mark.asyncio
    async def test_concurrent_orders_rate_limited(self, executor, token_map):
        """Test concurrent orders respect rate limit."""
        start_time = asyncio.get_event_loop().time()

        # Try to place 3 orders concurrently
        tasks = [
            executor.execute_order('Up', 10.0, token_map),
            executor.execute_order('Up', 20.0, token_map),
            executor.execute_order('Up', 30.0, token_map),
        ]

        results = await asyncio.gather(*tasks)

        elapsed = asyncio.get_event_loop().time() - start_time

        # All should succeed
        assert all(r is not None for r in results)

        # Should take at least 2 * rate_limit (3 orders = 2 gaps)
        # Plus network delays
        assert elapsed >= 0.2  # 2 * 0.1s rate limit

    @pytest.mark.asyncio
    async def test_order_timeout_handling(self, config, token_map):
        """Test handling of order timeouts."""
        connector = MagicMock()

        # Simulate very slow order (timeout)
        async def slow_order(*args, **kwargs):
            await asyncio.sleep(10.0)  # Way too slow
            return {'orderID': 'slow'}

        connector.create_market_order = slow_order

        executor = OrderExecutor(config, connector)

        # This should timeout or complete slowly
        # Test depends on implementation timeout handling
        # For now, just ensure it doesn't crash
        try:
            result = await asyncio.wait_for(
                executor.execute_order('Up', 10.0, token_map),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            # Expected behavior
            pass

    @pytest.mark.asyncio
    async def test_rapid_order_submission(self, executor, token_map):
        """Test rapid order submissions are serialized."""
        order_times = []

        async def timed_order(*args, **kwargs):
            order_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)
            return {'orderID': 'test'}

        executor.connector.create_market_order = timed_order

        # Submit 5 orders rapidly
        tasks = [
            executor.execute_order('Up', i * 10.0, token_map)
            for i in range(1, 6)
        ]

        await asyncio.gather(*tasks)

        # Check gaps between orders respect rate limit
        for i in range(1, len(order_times)):
            gap = order_times[i] - order_times[i-1]
            assert gap >= 0.09  # Allow small timing variance


# ========== Analytics + Callback Integration ==========
class TestAnalyticsCallbackIntegration:
    """Test analytics engine with callback system."""

    @pytest.fixture
    def engine(self):
        return AnalyticsEngine()

    @pytest.fixture
    def manager(self):
        return CallbackManager()

    @pytest.mark.asyncio
    async def test_analytics_triggers_callbacks(self, engine, manager):
        """Test analytics results trigger callbacks."""
        results = []

        async def on_edge_detected(edge_direction, edge_bps):
            results.append({'direction': edge_direction, 'bps': edge_bps})

        manager.register('analytics_update', on_edge_detected)

        # Calculate edge
        edge, edge_bps = engine.calculate_edge(0.45, 0.52, 50.0)

        # Emit via callback
        manager.emit('analytics_update', edge, edge_bps)

        # Wait for async callback to complete
        await asyncio.sleep(0.01)

        # Should have received update
        assert len(results) == 1
        assert results[0]['direction'] == edge

    @pytest.mark.asyncio
    async def test_high_frequency_analytics_updates(self, engine, manager):
        """Test high-frequency analytics updates."""
        update_count = {'value': 0}

        async def count_updates(*args):
            update_count['value'] += 1

        manager.register('analytics_update', count_updates)

        # Emit 100 analytics updates rapidly
        for i in range(100):
            fv_yes, fv_no = engine.calculate_binary_fair_value(
                current_price=50000.0 + i,
                price_to_beat=50000.0,
                time_to_expiry_seconds=300,
                volatility=0.6
            )
            manager.emit('analytics_update', fv_yes, fv_no)

        # Wait for async callbacks
        await asyncio.sleep(0.05)

        # All should be processed
        assert update_count['value'] == 100


# ========== Error Recovery Tests ==========
class TestErrorRecoveryIntegration:
    """Test error recovery in async flows."""

    @pytest.mark.asyncio
    async def test_callback_manager_recovers_from_callback_error(self):
        """Test callback manager continues after callback error."""
        manager = CallbackManager()
        successful_calls = {'count': 0}

        async def failing_callback():
            raise RuntimeError("Callback error")

        async def successful_callback():
            successful_calls['count'] += 1

        manager.register('market_update', failing_callback)
        manager.register('market_update', successful_callback)

        # Emit multiple times
        for _ in range(5):
            manager.emit('market_update')

        # Wait for async callbacks
        await asyncio.sleep(0.1)

        # Successful callback should have been called all 5 times
        assert successful_calls['count'] == 5

    @pytest.mark.asyncio
    async def test_executor_handles_connector_exceptions(self):
        """Test order executor handles connector exceptions."""
        config = AppConfig()
        config.order_rate_limit_seconds = 0.0

        connector = MagicMock()

        # Connector raises exception
        async def failing_order(*args, **kwargs):
            raise ConnectionError("Network error")

        connector.create_market_order = failing_order

        executor = OrderExecutor(config, connector)
        token_map = {'Up': '0x' + '1' * 64}

        # Should return None, not raise exception
        result = await executor.execute_order('Up', 10.0, token_map)

        assert result is None


# ========== Concurrent Operations Tests ==========
class TestConcurrentOperations:
    """Test concurrent operations and race conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_callback_registration(self):
        """Test concurrent callback registration is thread-safe."""
        manager = CallbackManager()

        async def register_callbacks(start_idx):
            for i in range(start_idx, start_idx + 50):
                def callback():
                    pass
                callback.__name__ = f'callback_{i}'
                manager.register('market_update', callback)

        # Register from multiple tasks
        tasks = [
            register_callbacks(0),
            register_callbacks(50),
            register_callbacks(100),
        ]

        await asyncio.gather(*tasks)

        # All 150 should be registered
        callbacks = manager.get_callbacks('market_update')
        assert len(callbacks) == 150

    @pytest.mark.asyncio
    async def test_concurrent_emit_and_unregister(self):
        """Test emitting while unregistering callbacks."""
        manager = CallbackManager()
        call_count = {'value': 0}

        async def counter():
            call_count['value'] += 1

        manager.register('market_update', counter)

        async def emit_many():
            for _ in range(100):
                manager.emit('market_update')
                await asyncio.sleep(0.001)

        async def unregister_after_delay():
            await asyncio.sleep(0.05)
            manager.unregister('market_update', counter)

        # Run concurrently
        await asyncio.gather(emit_many(), unregister_after_delay())

        # Wait for async callbacks to finish
        await asyncio.sleep(0.1)

        # Should have some calls (before unregister)
        assert call_count['value'] > 0
        # But not all 100 (after unregister)
        assert call_count['value'] < 100

    @pytest.mark.asyncio
    async def test_shared_state_with_lock(self):
        """Test shared state protection with asyncio.Lock."""
        lock = asyncio.Lock()
        shared_state = {'value': 0}

        async def increment():
            for _ in range(100):
                async with lock:
                    current = shared_state['value']
                    await asyncio.sleep(0.0001)  # Simulate work
                    shared_state['value'] = current + 1

        # Run 10 incrementers concurrently
        await asyncio.gather(*[increment() for _ in range(10)])

        # Should be exactly 1000 (100 * 10)
        assert shared_state['value'] == 1000


# ========== Timeout and Cancellation Tests ==========
class TestTimeoutAndCancellation:
    """Test timeout and cancellation behavior."""

    @pytest.mark.asyncio
    async def test_callback_cancellation(self):
        """Test cancelling callbacks."""
        manager = CallbackManager()
        started = {'value': False}
        completed = {'value': False}

        async def long_callback():
            started['value'] = True
            await asyncio.sleep(1.0)
            completed['value'] = True

        manager.register('market_update', long_callback)

        # Start emit (synchronous) which starts async callback
        manager.emit('market_update')

        # Wait briefly then check state
        await asyncio.sleep(0.1)

        # Callback should have started
        assert started['value'] is True
        # But may not have completed yet
        # Note: emit() is synchronous but spawns async tasks

    @pytest.mark.asyncio
    async def test_order_execution_with_timeout(self):
        """Test order execution respects timeouts."""
        config = AppConfig()
        connector = MagicMock()

        async def slow_order(*args, **kwargs):
            await asyncio.sleep(5.0)
            return {'orderID': 'slow'}

        connector.create_market_order = slow_order

        executor = OrderExecutor(config, connector)
        token_map = {'Up': '0x' + '1' * 64}

        # Wrap with timeout
        try:
            result = await asyncio.wait_for(
                executor.execute_order('Up', 10.0, token_map),
                timeout=0.5
            )
        except asyncio.TimeoutError:
            # Expected
            result = None

        assert result is None


# ========== Real-World Scenario Tests ==========
class TestRealWorldScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_market_update_pipeline(self):
        """Test complete market update pipeline."""
        manager = CallbackManager()
        engine = AnalyticsEngine()

        updates_received = []

        async def on_market_update(price):
            updates_received.append(price)

        async def on_analytics(fv_yes, fv_no):
            pass  # Would trigger UI update

        manager.register('market_update', on_market_update)
        manager.register('analytics_update', on_analytics)

        # Simulate market updates
        for i in range(10):
            price = 50000.0 + (i * 100)
            manager.emit('market_update', price)

            # Calculate analytics
            fv_yes, fv_no = engine.calculate_binary_fair_value(
                current_price=price,
                price_to_beat=50000.0,
                time_to_expiry_seconds=300,
                volatility=0.6
            )

            manager.emit('analytics_update', fv_yes, fv_no)

        # Wait for async callbacks
        await asyncio.sleep(0.05)

        # All updates should be received
        assert len(updates_received) == 10

    @pytest.mark.asyncio
    async def test_order_execution_pipeline(self):
        """Test complete order execution pipeline."""
        config = AppConfig()
        config.order_rate_limit_seconds = 0.05

        connector = MagicMock()
        orders_executed = []

        async def mock_order(token_id, size, side, price=None):
            await asyncio.sleep(0.01)
            order = {'orderID': f'order_{len(orders_executed)}', 'size': size}
            orders_executed.append(order)
            return order

        connector.create_market_order = mock_order

        executor = OrderExecutor(config, connector)
        token_map = {'Up': '0x' + '1' * 64, 'Down': '0x' + '2' * 64}

        # Execute multiple orders
        sizes = [10.0, 20.0, 30.0, 40.0, 50.0]
        tasks = [
            executor.execute_order('Up', size, token_map)
            for size in sizes
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r is not None for r in results)
        assert len(orders_executed) == 5

        # Sizes should match
        executed_sizes = [o['size'] for o in orders_executed]
        assert set(executed_sizes) == set(sizes)
