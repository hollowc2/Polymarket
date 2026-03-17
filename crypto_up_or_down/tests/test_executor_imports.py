import importlib

from polymarket_algo.core import DataFeed, PriceTick
from polymarket_algo.executor import LiveTrader, TradingState
from polymarket_algo.executor.feed import PolymarketDataFeed


def test_executor_submodules_import() -> None:
    modules = [
        "polymarket_algo.executor",
        "polymarket_algo.executor.blockchain",
        "polymarket_algo.executor.client",
        "polymarket_algo.executor.feed",
        "polymarket_algo.executor.resilience",
        "polymarket_algo.executor.trader",
        "polymarket_algo.executor.ws",
    ]
    for module in modules:
        importlib.import_module(module)


def test_polymarket_data_feed_conforms_protocol() -> None:
    feed = PolymarketDataFeed()
    assert isinstance(feed, DataFeed)


def test_data_feed_protocol_importable_from_core() -> None:
    # Imported at module level; this verifies re-export and symbol availability.
    assert DataFeed is not None


def test_price_tick_importable_and_constructable() -> None:
    tick = PriceTick(symbol="btc-up", price=0.62, timestamp=1234567890.0, size=10.0, side="buy", source="test")
    assert tick.symbol == "btc-up"
    assert tick.price == 0.62
    assert tick.timestamp == 1234567890.0
    assert tick.size == 10.0
    assert tick.side == "buy"
    assert tick.source == "test"


def test_data_feed_protocol_has_required_methods() -> None:
    for name in ("subscribe", "unsubscribe", "on_tick", "on_reconnect"):
        assert hasattr(DataFeed, name)


def test_executor_package_exports_live_trader_and_trading_state() -> None:
    assert LiveTrader is not None
    assert TradingState is not None
