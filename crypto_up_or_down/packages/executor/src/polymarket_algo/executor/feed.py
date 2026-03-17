import time
from collections.abc import Callable

from polymarket_algo.core.types import PriceTick
from polymarket_algo.executor.ws import PolymarketWebSocket, TradeEvent


class PolymarketDataFeed:
    """Thin DataFeed adapter around ``PolymarketWebSocket``."""

    name = "polymarket-websocket"

    def __init__(self):
        self._tick_callbacks: list[Callable[[PriceTick], None]] = []
        self._reconnect_callbacks: list[Callable[[], None]] = []
        self._ws = PolymarketWebSocket(on_trade=self._handle_trade)
        self._ws.set_mid_change_callback(self._handle_mid_change)

    def start(self) -> None:
        self._ws.start()

    def stop(self) -> None:
        self._ws.stop()

    def subscribe(self, symbol: str, **kwargs) -> None:
        self._ws.subscribe_market(symbol, token_ids=kwargs.get("token_ids"))

    def unsubscribe(self, symbol: str) -> None:
        self._ws.unsubscribe_market(symbol)

    def on_tick(self, callback: Callable[[PriceTick], None]) -> None:
        self._tick_callbacks.append(callback)

    def on_reconnect(self, callback: Callable[[], None]) -> None:
        self._reconnect_callbacks.append(callback)

    def is_connected(self) -> bool:
        return self._ws.is_connected()

    def subscribe_market(self, condition_id: str, token_ids: list[str] | None = None) -> None:
        """Backward-compatible alias for direct market subscriptions."""
        self.subscribe(condition_id, token_ids=token_ids)

    def unsubscribe_market(self, condition_id: str) -> None:
        """Backward-compatible alias for direct market unsubscriptions."""
        self.unsubscribe(condition_id)

    def _emit_tick(self, tick: PriceTick) -> None:
        for callback in self._tick_callbacks:
            callback(tick)

    def _handle_trade(self, trade: TradeEvent) -> None:
        symbol = trade.market_id or trade.token_id
        self._emit_tick(
            PriceTick(
                symbol=symbol,
                price=trade.price,
                timestamp=trade.timestamp,
                size=trade.size,
                side=trade.side.lower() if trade.side else None,
                source="polymarket-trade",
            )
        )

    def _handle_mid_change(self, token_id: str, mid_price: float) -> None:
        self._emit_tick(
            PriceTick(
                symbol=token_id,
                price=mid_price,
                timestamp=time.time(),
                source="polymarket-mid",
            )
        )
