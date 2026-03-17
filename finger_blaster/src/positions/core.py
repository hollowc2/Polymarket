"""
PositionsCore - Main orchestrator for Position Manager.

Provides:
- Position monitoring via Data API polling
- Market discovery for 15m BTC filter
- Position closing via market orders
- Event-driven architecture with AsyncEventBus
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from src.connectors.polymarket import PolymarketConnector
from src.positions.config import PositionsConfig
from src.shared.market_discovery import MarketDiscoveryConfig, MarketDiscoveryService

logger = logging.getLogger("FingerBlaster.Positions")
logger.setLevel(logging.DEBUG)  # Ensure DEBUG level for detailed logs

# WebSocket constants for CLOB real-time price feed
_WS_URI = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_WS_PING_INTERVAL = 20
_WS_PING_TIMEOUT = 10
_WS_RECV_TIMEOUT = 30.0
_WS_RECONNECT_DELAY = 5
_WS_MAX_RECONNECT_ATTEMPTS = 10
_MAX_WS_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
_PRICE_EMIT_THROTTLE = 0.2  # Minimum seconds between UI price updates


@dataclass
class Position:
    """Represents a single position."""

    token_id: str
    condition_id: str
    outcome: str  # "Up" or "Down" (or legacy "Yes"/"No")
    size: float
    avg_entry_price: float
    current_price: float
    initial_value: float
    current_value: float
    pnl_usd: float
    pnl_pct: float
    market_title: str = ""
    is_current_market: bool = False


class AsyncEventBus:
    """Simple async event bus for UI callbacks."""

    def __init__(self):
        self._subs: Dict[str, List[Callable]] = {}

    def on(self, event: str, cb: Callable) -> None:
        """Register a callback for an event."""
        self._subs.setdefault(event, []).append(cb)

    def off(self, event: str, cb: Callable) -> None:
        """Unregister a callback."""
        if event in self._subs and cb in self._subs[event]:
            self._subs[event].remove(cb)

    def emit(self, event: str, *args) -> None:
        """Emit an event to all registered callbacks."""
        for cb in self._subs.get(event, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(*args))
                else:
                    cb(*args)
            except Exception as e:
                logger.exception(f"Event handler error for {event}: {e}")


class PositionsCore:
    """
    Core orchestrator for Position Manager.

    Monitors wallet positions, provides filtering for current 15m market,
    and handles position closing.

    Events emitted:
    - 'positions_update': List[Position] - when positions change
    - 'market_update': Dict[str, Any] - current 15m market info
    - 'filter_changed': bool - filter mode toggled
    - 'position_closed': str, bool - token_id, success
    - 'connection_status': bool, str - connected, message
    - 'log': str - log message
    """

    def __init__(self, config: Optional[PositionsConfig] = None):
        self.config = config or PositionsConfig()
        self.bus = AsyncEventBus()

        self._connector: Optional[PolymarketConnector] = None
        self._market_discovery: Optional[MarketDiscoveryService] = None
        self._positions: List[Position] = []
        self._filter_current_market: bool = False
        self._running: bool = False

        self._position_poll_task: Optional[asyncio.Task] = None
        self._current_market: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()

        # WebSocket price feed state
        self._order_books: Dict[str, Dict[str, Dict[float, float]]] = {}  # token_id -> {bids, asks}
        self._live_prices: Dict[str, float] = {}  # token_id -> live mid price
        self._subscribed_token_ids: Set[str] = set()
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_shutdown = asyncio.Event()
        self._last_price_emit_time: float = 0.0

    def on(self, event: str, cb: Callable) -> None:
        """Register a callback for an event."""
        self.bus.on(event, cb)

    def off(self, event: str, cb: Callable) -> None:
        """Unregister a callback."""
        self.bus.off(event, cb)

    async def start(self) -> None:
        """Start the position manager."""
        if self._running:
            logger.debug("PositionsCore already running")
            return

        self._running = True
        logger.info("PositionsCore starting...")

        # Initialize connector
        self._connector = PolymarketConnector()
        await self._connector._ensure_async_session()

        # Initialize market discovery (uses connector for single source of truth)
        discovery_config = MarketDiscoveryConfig(
            series_id=self.config.btc_series_id,
            poll_interval=self.config.market_poll_interval,
        )
        logger.info(
            f"Initializing market discovery for series {self.config.btc_series_id} "
            f"with poll interval {self.config.market_poll_interval}s"
        )
        self._market_discovery = MarketDiscoveryService(
            connector=self._connector,
            config=discovery_config,
            on_market_change=self._on_market_change,
        )
        await self._market_discovery.start()

        # Wait a moment for initial market discovery
        await asyncio.sleep(1)
        current_market = await self._market_discovery.get_current_market()
        if current_market:
            logger.warning(f"Initial market found: {current_market.get('market_id')} - {current_market.get('title', 'Unknown')}")
            # Emit initial market to UI
            logger.warning("Emitting initial market update to UI")
            self.bus.emit("market_update", current_market)
            async with self._lock:
                self._current_market = current_market
        else:
            logger.error("No active market found during startup")

        # Start position polling
        self._position_poll_task = asyncio.create_task(self._position_poll_loop())

        self.bus.emit("connection_status", True, "Connected")
        self.bus.emit("log", "Position Manager started")
        logger.info("PositionsCore started")

    async def stop(self) -> None:
        """Stop the position manager."""
        if not self._running:
            return

        self._running = False
        logger.info("PositionsCore stopping...")

        # Stop WebSocket price feed
        await self._stop_ws()

        # Stop polling
        if self._position_poll_task:
            self._position_poll_task.cancel()
            try:
                await self._position_poll_task
            except asyncio.CancelledError:
                pass
            self._position_poll_task = None

        # Stop market discovery
        if self._market_discovery:
            await self._market_discovery.stop()
            self._market_discovery = None

        # Close connector
        if self._connector:
            await self._connector.close()
            self._connector = None

        self.bus.emit("connection_status", False, "Disconnected")
        logger.info("PositionsCore stopped")

    def toggle_filter(self) -> bool:
        """Toggle filter between all positions and current market only."""
        self._filter_current_market = not self._filter_current_market
        self.bus.emit("filter_changed", self._filter_current_market)
        self.bus.emit(
            "log",
            f"Filter: {'Current 15m Market' if self._filter_current_market else 'All Positions'}",
        )
        # Re-emit positions with new filter
        self._emit_filtered_positions()
        return self._filter_current_market

    def get_filter_state(self) -> bool:
        """Get current filter state."""
        return self._filter_current_market

    async def close_position(self, token_id: str) -> bool:
        """
        Close a position by selling all shares.

        Args:
            token_id: Token ID of position to close

        Returns:
            True if successful, False otherwise
        """
        if not self._connector:
            logger.error("Connector not initialized")
            return False

        try:
            self.bus.emit("log", f"Closing position {token_id[:20]}...")

            # Get balance
            balance = await self._connector.get_token_balance(token_id)
            if balance < self.config.min_position_size:
                self.bus.emit("log", f"Position already closed or too small: {balance}")
                self.bus.emit("position_closed", token_id, True)
                return True

            # Create market sell order
            result = await self._connector.create_market_order(
                token_id=token_id,
                amount=balance,
                side="SELL",
            )

            if result and isinstance(result, dict) and result.get("orderID"):
                self.bus.emit("log", f"Position closed: {result.get('orderID')}")
                self.bus.emit("position_closed", token_id, True)
                # Force position refresh
                asyncio.create_task(self._fetch_positions())
                return True
            else:
                self.bus.emit("log", f"Failed to close position: {result}")
                self.bus.emit("position_closed", token_id, False)
                return False

        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            self.bus.emit("log", f"Error closing position: {e}")
            self.bus.emit("position_closed", token_id, False)
            return False

    async def refresh_positions(self) -> None:
        """Force refresh of positions."""
        await self._fetch_positions()

    async def get_positions(self) -> List[Position]:
        """Get current positions (with filter applied)."""
        async with self._lock:
            return self._get_filtered_positions()

    async def get_current_market(self) -> Optional[Dict[str, Any]]:
        """Get the current 15m BTC market."""
        if self._market_discovery:
            return await self._market_discovery.get_current_market()
        return None

    async def _on_market_change(self, market: Dict[str, Any]) -> None:
        """Handle market change from discovery service."""
        logger.warning(f"PositionsCore._on_market_change called with market: {market.get('market_id', 'Unknown')}")

        async with self._lock:
            self._current_market = market

        market_id = market.get('market_id', 'Unknown')
        token_map = market.get('token_map', {})
        logger.warning(
            f"PositionsCore: Market changed to {market_id}: {market.get('title', 'Unknown')}"
        )
        logger.info(f"New market token map: {token_map}")

        logger.warning("PositionsCore: Emitting market_update event to UI")
        self.bus.emit("market_update", market)
        self.bus.emit(
            "log",
            f"Market: {market.get('title', 'Unknown')[:50]}",
        )

        # Re-tag positions with current market flag
        logger.info("Re-tagging positions after market change")
        await self._fetch_positions()

    async def _position_poll_loop(self) -> None:
        """Polling loop for position updates."""
        while self._running:
            try:
                await self._fetch_positions()
            except Exception as e:
                logger.error(f"Error in position poll loop: {e}", exc_info=True)

            await asyncio.sleep(self.config.position_poll_interval)

    async def _fetch_positions(self) -> None:
        """Fetch positions from Data API."""
        if not self._connector:
            logger.warning("Connector not initialized in _fetch_positions")
            return

        try:
            logger.debug(f"Fetching positions with threshold {self.config.min_position_size}")
            raw_positions = await self._connector.get_positions(
                size_threshold=self.config.min_position_size
            )

            # None means the API call itself failed — keep displaying whatever
            # we already have rather than wiping the UI on a transient error.
            if raw_positions is None:
                logger.warning("Position API call failed, keeping current positions")
                return

            logger.debug(f"Received {len(raw_positions)} raw positions from API")

            # Get current market token IDs for tagging
            current_token_ids = set()
            if self._market_discovery:
                token_map = await self._market_discovery.get_token_map()
                current_token_ids = set(token_map.values())
                logger.debug(f"Current market token map: {token_map}")
                logger.debug(f"Current market token IDs: {current_token_ids}")

            # Convert to Position objects
            positions = []
            filtered_count = 0
            for raw in raw_positions:
                try:
                    position = self._parse_position(raw, current_token_ids)
                    if position:
                        positions.append(position)
                    else:
                        filtered_count += 1
                except Exception as e:
                    logger.warning(f"Error parsing position: {e}")

            logger.debug(f"Parsed {len(positions)} positions, filtered {filtered_count}")
            logger.debug(f"Positions matching current market: {sum(1 for p in positions if p.is_current_market)}")

            # Update positions
            async with self._lock:
                self._positions = positions

            # Restart WebSocket if the set of tracked tokens changed
            new_token_ids = {p.token_id for p in positions}
            if new_token_ids != self._subscribed_token_ids:
                self._subscribed_token_ids = new_token_ids
                # Prune cached prices for tokens no longer held
                self._live_prices = {
                    tid: price for tid, price in self._live_prices.items()
                    if tid in new_token_ids
                }
                await self._start_ws()

            # Emit update
            self._emit_filtered_positions()

        except Exception as e:
            logger.error(f"Error fetching positions: {e}", exc_info=True)

    def _parse_position(
        self, raw: Dict[str, Any], current_token_ids: set
    ) -> Optional[Position]:
        """
        Parse raw position data into Position object.

        Args:
            raw: Raw position dict from API
            current_token_ids: Set of token IDs for current 15m market

        Returns:
            Position object or None if invalid
        """
        token_id = raw.get("asset", "")
        if not token_id:
            return None

        size = float(raw.get("size", 0))

        # Check if this is a current market position being filtered by size
        if token_id in current_token_ids and size < self.config.min_position_size:
            logger.warning(
                f"Current market position filtered by size threshold: "
                f"size={size:.4f}, threshold={self.config.min_position_size}, "
                f"token={token_id[:20]}..."
            )

        if size < self.config.min_position_size:
            return None

        # Parse outcome (Up/Down or Yes/No)
        outcome = raw.get("outcome", "")
        if not outcome:
            # Try to infer from proxyTicker
            proxy_ticker = raw.get("proxyTicker", "")
            if "up" in proxy_ticker.lower():
                outcome = "Up"
            elif "down" in proxy_ticker.lower():
                outcome = "Down"
            elif "yes" in proxy_ticker.lower():
                outcome = "Yes"
            elif "no" in proxy_ticker.lower():
                outcome = "No"
            else:
                outcome = "Unknown"

        # Normalize outcome to Up/Down
        if outcome.upper() in ("YES", "UP"):
            outcome = "Up"
        elif outcome.upper() in ("NO", "DOWN"):
            outcome = "Down"

        avg_entry = float(raw.get("avgPrice", 0))
        current_price = float(raw.get("curPrice", 0))
        initial_value = float(raw.get("initialValue", 0))
        current_value = float(raw.get("currentValue", 0))
        pnl_usd = float(raw.get("cashPnl", 0))
        pnl_pct = float(raw.get("percentPnl", 0))

        # Check if this is current market
        is_current = token_id in current_token_ids

        # Get market title if available
        market_title = raw.get("title", "") or raw.get("marketTitle", "")

        # Debug logging for position tagging
        if current_token_ids:
            logger.debug(
                f"Position {outcome} (token={token_id[:20]}...) "
                f"is_current={is_current} title={market_title[:40]}"
            )

        return Position(
            token_id=token_id,
            condition_id=raw.get("conditionId", ""),
            outcome=outcome,
            size=size,
            avg_entry_price=avg_entry,
            current_price=current_price,
            initial_value=initial_value,
            current_value=current_value,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            market_title=market_title,
            is_current_market=is_current,
        )

    def _get_filtered_positions(self) -> List[Position]:
        """Get positions with current filter applied."""
        if self._filter_current_market:
            return [p for p in self._positions if p.is_current_market]
        return list(self._positions)

    def _emit_filtered_positions(self) -> None:
        """Emit positions update with live price overlay and recalculated PnL."""
        filtered = self._get_filtered_positions()

        # Overlay live mid prices from WebSocket and recalculate PnL
        for pos in filtered:
            live_price = self._live_prices.get(pos.token_id)
            if live_price is not None:
                pos.current_price = live_price
                pos.current_value = pos.size * live_price
                pos.pnl_usd = pos.current_value - pos.initial_value
                pos.pnl_pct = (pos.pnl_usd / pos.initial_value * 100.0) if pos.initial_value > 0 else 0.0

        self.bus.emit("positions_update", filtered)

    # ─── WebSocket price feed ─────────────────────────────────────────────

    async def _start_ws(self) -> None:
        """Start or restart the WebSocket price feed for current position tokens."""
        await self._stop_ws()
        if self._subscribed_token_ids:
            self._ws_shutdown = asyncio.Event()
            self._order_books = {}  # Fresh order books for new subscription
            self._ws_task = asyncio.create_task(self._ws_loop())
            logger.info(f"Price feed WebSocket starting for {len(self._subscribed_token_ids)} tokens")

    async def _stop_ws(self) -> None:
        """Stop the WebSocket price feed."""
        self._ws_shutdown.set()
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

    async def _ws_loop(self) -> None:
        """WebSocket connection loop with auto-reconnect for real-time order book data."""
        reconnect_attempts = 0

        while not self._ws_shutdown.is_set() and reconnect_attempts < _WS_MAX_RECONNECT_ATTEMPTS:
            token_ids = list(self._subscribed_token_ids)
            if not token_ids:
                break

            try:
                async with connect(
                    _WS_URI,
                    ping_interval=_WS_PING_INTERVAL,
                    ping_timeout=_WS_PING_TIMEOUT,
                    max_size=_MAX_WS_MESSAGE_SIZE,
                ) as ws:
                    await ws.send(json.dumps({"assets_ids": token_ids, "type": "market"}))
                    reconnect_attempts = 0
                    logger.info(f"Price feed connected, subscribed to {len(token_ids)} tokens")

                    while not self._ws_shutdown.is_set():
                        # Reconnect if the token set changed underneath us
                        if set(token_ids) != self._subscribed_token_ids:
                            logger.info("Token set changed, price feed reconnecting")
                            break

                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=_WS_RECV_TIMEOUT)
                            if len(message) > _MAX_WS_MESSAGE_SIZE:
                                continue
                            data = json.loads(message)
                            items = data if isinstance(data, list) else [data]
                            for item in items:
                                self._process_ws_message(item)
                        except asyncio.TimeoutError:
                            continue
                        except ConnectionClosed:
                            logger.warning("Price feed WebSocket closed")
                            break
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Price feed recv error: {e}")
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                reconnect_attempts += 1
                if reconnect_attempts >= _WS_MAX_RECONNECT_ATTEMPTS:
                    logger.error("Price feed: max reconnect attempts reached")
                    break
                wait_time = _WS_RECONNECT_DELAY * (2 ** min(reconnect_attempts - 1, 3))
                logger.info(f"Price feed reconnecting in {wait_time}s (attempt {reconnect_attempts})")
                try:
                    await asyncio.sleep(wait_time)
                except asyncio.CancelledError:
                    break

    def _process_ws_message(self, item: Dict[str, Any]) -> None:
        """Process a single WebSocket order book message, update mid price, throttle UI emit."""
        if not isinstance(item, dict):
            return

        asset_id = item.get('asset_id')
        if not asset_id or asset_id not in self._subscribed_token_ids:
            return

        if asset_id not in self._order_books:
            self._order_books[asset_id] = {'bids': {}, 'asks': {}}
        book = self._order_books[asset_id]

        if 'bids' in item and 'asks' in item:
            # Full order book snapshot
            book['bids'] = {
                float(x['price']): float(x['size'])
                for x in item['bids']
                if isinstance(x, dict) and 'price' in x and 'size' in x
            }
            book['asks'] = {
                float(x['price']): float(x['size'])
                for x in item['asks']
                if isinstance(x, dict) and 'price' in x and 'size' in x
            }
        elif 'price_changes' in item:
            # Incremental delta
            for change in item.get('price_changes', []):
                if not isinstance(change, dict):
                    continue
                try:
                    price = float(change.get('price', 0))
                    size = float(change.get('size', 0))
                    side = str(change.get('side', '')).upper()
                    if side == 'BUY':
                        if size <= 0:
                            book['bids'].pop(price, None)
                        else:
                            book['bids'][price] = size
                    elif side == 'SELL':
                        if size <= 0:
                            book['asks'].pop(price, None)
                        else:
                            book['asks'][price] = size
                except (ValueError, TypeError):
                    continue
        else:
            return

        # Mid price from best bid / best ask
        best_bid = max(book['bids']) if book['bids'] else 0.0
        best_ask = min(book['asks']) if book['asks'] else 1.0

        if best_bid > 0 and best_ask < 1.0:
            mid = (best_bid + best_ask) / 2.0
        elif best_ask < 1.0:
            mid = best_ask
        elif best_bid > 0:
            mid = best_bid
        else:
            return  # Cannot determine price

        self._live_prices[asset_id] = mid

        # Throttled UI update to avoid flooding the renderer
        now = time.time()
        if now - self._last_price_emit_time >= _PRICE_EMIT_THROTTLE:
            self._last_price_emit_time = now
            self._emit_filtered_positions()
