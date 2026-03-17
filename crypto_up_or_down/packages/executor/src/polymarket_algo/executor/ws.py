"""WebSocket client for real-time Polymarket data feeds.

Provides ~100ms latency vs ~1s for REST API polling.
"""

import asyncio
import json
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import urlparse

import websockets
from polymarket_algo.executor.client import DelayImpactModel, PolymarketClient
from websockets.exceptions import ConnectionClosed


@asynccontextmanager
async def _ws_connect(url: str, **kwargs):
    """Connect to a WebSocket, routing through HTTPS_PROXY/HTTP_PROXY if set.

    Supports socks5h://, socks5://, and http:// proxy URLs.
    Requires python-socks[asyncio] for SOCKS5 proxies.
    Falls back to a direct connection if python-socks is not installed.
    """
    proxy_url = (
        os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or ""
    )
    if proxy_url:
        try:
            from python_socks.async_.asyncio import Proxy

            parsed = urlparse(url)
            host = parsed.hostname or url
            port = parsed.port or (443 if url.startswith("wss://") else 80)
            proxy = Proxy.from_url(proxy_url)
            sock = await proxy.connect(dest_host=host, dest_port=port)
            async with websockets.connect(url, sock=sock, **kwargs) as ws:
                yield ws
            return
        except ImportError:
            print("[ws] python-socks not installed — connecting without proxy")
    async with websockets.connect(url, **kwargs) as ws:
        yield ws


@dataclass
class OrderBookLevel:
    """Single price level in orderbook."""

    price: float
    size: float


@dataclass
class CachedOrderBook:
    """Cached order book state with timestamp."""

    token_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid: float = 0.5

    def update_from_snapshot(self, data: dict):
        """Update from full orderbook snapshot."""
        self.bids = [OrderBookLevel(float(b["price"]), float(b["size"])) for b in data.get("bids", [])]
        self.asks = [OrderBookLevel(float(a["price"]), float(a["size"])) for a in data.get("asks", [])]
        self._recalculate()

    def update_from_delta(self, data: dict):
        """Update from orderbook delta (price_change event)."""
        changes = data.get("changes", [])
        for change in changes:
            side = change.get("side")
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))

            if side == "BUY":
                self._update_level(self.bids, price, size, reverse=True)
            elif side == "SELL":
                self._update_level(self.asks, price, size, reverse=False)
        self._recalculate()

    def _update_level(self, levels: list[OrderBookLevel], price: float, size: float, reverse: bool):
        """Update a single price level."""
        # Find existing level
        for i, level in enumerate(levels):
            if abs(level.price - price) < 0.0001:
                if size == 0:
                    levels.pop(i)
                else:
                    level.size = size
                return

        # Add new level if size > 0
        if size > 0:
            levels.append(OrderBookLevel(price, size))
            levels.sort(key=lambda x: x.price, reverse=reverse)

    def _recalculate(self):
        """Recalculate best bid/ask and mid."""
        self.timestamp = time.time()
        if self.bids:
            self.bids.sort(key=lambda x: x.price, reverse=True)
            self.best_bid = self.bids[0].price
        if self.asks:
            self.asks.sort(key=lambda x: x.price)
            self.best_ask = self.asks[0].price
        if self.best_bid > 0 and self.best_ask > 0:
            self.mid = (self.best_bid + self.best_ask) / 2

    def get_execution_price(self, side: str, amount_usd: float) -> tuple[float, float, float]:
        """Calculate execution price by walking the book.

        Returns: (execution_price, slippage_pct, fill_pct)
        """
        levels = self.asks if side == "BUY" else self.bids

        if not levels:
            return self.mid, 0.0, 0.0

        remaining = amount_usd
        total_shares = 0.0
        total_cost = 0.0

        for level in levels:
            if remaining <= 0:
                break
            level_value = level.price * level.size
            if level_value >= remaining:
                shares = remaining / level.price
                total_shares += shares
                total_cost += remaining
                remaining = 0
            else:
                total_shares += level.size
                total_cost += level_value
                remaining -= level_value

        if total_shares == 0:
            return self.mid, 0.0, 0.0

        exec_price = total_cost / total_shares
        filled_amount = amount_usd - remaining
        fill_pct = (filled_amount / amount_usd * 100) if amount_usd > 0 else 100.0

        # Calculate slippage vs best price
        best_price = self.best_ask if side == "BUY" else self.best_bid
        if best_price > 0:
            slippage_pct = abs(exec_price - best_price) / best_price * 100
        else:
            slippage_pct = 0.0

        return exec_price, slippage_pct, fill_pct


@dataclass
class TradeEvent:
    """Real-time trade event from WebSocket."""

    token_id: str
    market_id: str
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    timestamp: float  # unix seconds
    taker_address: str = ""
    maker_address: str = ""


class PolymarketWebSocket:
    """WebSocket client for real-time Polymarket data.

    Supports:
    - Order book subscriptions with ~100ms latency
    - Trade event streaming
    - Automatic reconnection
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    USER_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    def __init__(
        self,
        on_trade: Callable[[TradeEvent], None] | None = None,
        on_mid_change: Callable[[str, float], None] | None = None,
    ):
        """Initialize WebSocket client.

        Args:
            on_trade: Callback for trade events (called from asyncio thread)
            on_mid_change: Callback for orderbook midpoint changes (token_id, mid_price)
        """
        self._on_trade = on_trade
        self._on_mid_change = on_mid_change
        self._orderbooks: dict[str, CachedOrderBook] = {}
        self._subscribed_tokens: set[str] = set()
        self._subscribed_markets: set[str] = set()  # condition IDs
        self._ws = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._lock = threading.Lock()

        # Trade callback queue for thread-safe delivery
        self._trade_queue: list[TradeEvent] = []

        # Connection stats
        self.reconnect_count = 0
        self.last_message_time = 0.0
        self.messages_received = 0

    def start(self):
        """Start WebSocket connection in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Wait for connection (with timeout)
        timeout = 5.0
        start = time.time()
        while not self._connected.is_set() and time.time() - start < timeout:
            time.sleep(0.1)

    def stop(self):
        """Stop WebSocket connection gracefully."""
        self._running = False

        if self._loop and self._loop.is_running():
            try:
                # Schedule graceful shutdown
                asyncio.run_coroutine_threadsafe(self._graceful_shutdown(), self._loop)
                # Give it time to close cleanly
                time.sleep(0.5)
                # Now stop the loop
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                # Event loop already closed, ignore
                pass

        if self._thread:
            self._thread.join(timeout=2.0)

    async def _graceful_shutdown(self):
        """Gracefully close WebSocket and cancel tasks."""
        # Close WebSocket connection
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

        # Cancel all pending tasks except current
        tasks = [t for t in asyncio.all_tasks(self._loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()

        # Wait for tasks to be cancelled
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _run_loop(self):
        """Run asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            if self._running:  # Only log if not intentionally stopped
                print(f"[ws] Event loop error: {e}")
        finally:
            # Clean up any remaining tasks
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()

    async def _connect_loop(self):
        """Main connection loop with reconnection logic."""
        while self._running:
            try:
                async with _ws_connect(
                    self.WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    print(f"[ws] Connected to {self.WS_URL}")

                    # Resubscribe to any existing subscriptions
                    await self._resubscribe()

                    # Message handling loop
                    async for message in ws:
                        self.last_message_time = time.time()
                        self.messages_received += 1
                        await self._handle_message(message)

            except ConnectionClosed as e:
                print(f"[ws] Connection closed: {e}")
                self._connected.clear()
            except Exception as e:
                print(f"[ws] Connection error: {e}")
                self._connected.clear()

            if self._running:
                self.reconnect_count += 1
                wait_time = min(30, 2 ** min(self.reconnect_count, 5))
                print(f"[ws] Reconnecting in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def _resubscribe(self):
        """Resubscribe to all tokens/markets after reconnect."""
        with self._lock:
            markets = list(self._subscribed_markets)

        for market_id in markets:
            await self._send_subscribe(market_id)

    async def _send_subscribe(self, market_id: str):
        """Send subscription message for a market."""
        if not self._ws:
            return

        msg = {
            "type": "subscribe",
            "channel": "market",
            "market": market_id,
        }
        await self._ws.send(json.dumps(msg))

    async def _handle_message(self, raw: str | bytes):
        """Handle incoming WebSocket message."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", data.get("event_type", ""))

        if msg_type == "book":
            # Full orderbook snapshot
            token_id = data.get("asset_id", "")
            if token_id:
                with self._lock:
                    if token_id not in self._orderbooks:
                        self._orderbooks[token_id] = CachedOrderBook(token_id=token_id)
                    self._orderbooks[token_id].update_from_snapshot(data)

        elif msg_type == "price_change":
            # Orderbook delta
            token_id = data.get("asset_id", "")
            if token_id and token_id in self._orderbooks:
                mid_price: float | None = None
                with self._lock:
                    self._orderbooks[token_id].update_from_delta(data)
                    mid_price = self._orderbooks[token_id].mid

                if self._on_mid_change and mid_price is not None:
                    self._on_mid_change(token_id, mid_price)

        elif msg_type == "last_trade_price":
            # Trade event
            token_id = data.get("asset_id", "")
            market_id = data.get("market", "")
            price = float(data.get("price", 0))
            size = float(data.get("size", 0))
            side = data.get("side", "BUY")
            ts = float(data.get("timestamp", time.time()))

            trade = TradeEvent(
                token_id=token_id,
                market_id=market_id,
                price=price,
                size=size,
                side=side,
                timestamp=ts,
            )

            if self._on_trade:
                self._on_trade(trade)

    def subscribe_market(self, condition_id: str, token_ids: list[str] | None = None):
        """Subscribe to a market's orderbook and trade updates.

        Args:
            condition_id: Market condition ID
            token_ids: Optional list of token IDs to track orderbooks for
        """
        with self._lock:
            self._subscribed_markets.add(condition_id)

            if token_ids:
                for tid in token_ids:
                    self._subscribed_tokens.add(tid)
                    if tid not in self._orderbooks:
                        self._orderbooks[tid] = CachedOrderBook(token_id=tid)

        if self._loop and self._connected.is_set():
            asyncio.run_coroutine_threadsafe(self._send_subscribe(condition_id), self._loop)

    def unsubscribe_market(self, condition_id: str):
        """Unsubscribe from a market."""
        with self._lock:
            self._subscribed_markets.discard(condition_id)

        if self._loop and self._ws:
            msg = {
                "type": "unsubscribe",
                "channel": "market",
                "market": condition_id,
            }
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(msg)), self._loop)

    def get_orderbook(self, token_id: str) -> CachedOrderBook | None:
        """Get cached orderbook for a token.

        Returns None if not subscribed or no data yet.
        """
        with self._lock:
            return self._orderbooks.get(token_id)

    def get_execution_price(
        self, token_id: str, side: str, amount_usd: float, copy_delay_ms: int = 0
    ) -> tuple[float, float, float, float, float, dict | None]:
        """Get execution price from cached orderbook.

        Returns: (exec_price, spread, slippage_pct, fill_pct, delay_impact_pct, delay_breakdown)
        Falls back to REST API if no cached data.
        """
        book = self.get_orderbook(token_id)

        if book and book.timestamp > 0:
            # Use cached orderbook
            exec_price, slippage_pct, fill_pct = book.get_execution_price(side, amount_usd)
            spread = book.best_ask - book.best_bid if book.best_ask > 0 and book.best_bid > 0 else 0

            # Calculate depth at best level
            if side == "BUY":
                depth_at_best = book.asks[0].price * book.asks[0].size if book.asks else 0
            else:
                depth_at_best = book.bids[0].price * book.bids[0].size if book.bids else 0

            # Calculate delay impact using the improved model
            delay_impact_pct = 0.0
            delay_breakdown = None

            if copy_delay_ms > 0:
                delay_model = DelayImpactModel()
                delay_impact_pct, delay_breakdown = delay_model.calculate_impact(
                    delay_ms=copy_delay_ms,
                    order_size=amount_usd,
                    depth_at_best=depth_at_best,
                    spread=spread,
                    side=side,
                )

                if side == "BUY":
                    exec_price *= 1 + delay_impact_pct / 100
                else:
                    exec_price *= 1 - delay_impact_pct / 100
                exec_price = max(0.01, min(0.99, exec_price))

            return exec_price, spread, slippage_pct, fill_pct, delay_impact_pct, delay_breakdown

        # No cached data
        return 0.5, 0.0, 0.0, 100.0, 0.0, None

    def get_mid(self, token_id: str) -> float | None:
        """Get midpoint price from cached orderbook."""
        book = self.get_orderbook(token_id)
        if book and book.timestamp > 0:
            return book.mid
        return None

    def set_mid_change_callback(self, callback: Callable[[str, float], None] | None):
        """Set callback for orderbook midpoint updates."""
        self._on_mid_change = callback

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected.is_set()

    @property
    def stats(self) -> dict:
        """Get connection statistics."""
        with self._lock:
            subscribed_markets = len(self._subscribed_markets)
            cached_orderbooks = len(self._orderbooks)

        return {
            "connected": self.is_connected(),
            "reconnect_count": self.reconnect_count,
            "messages_received": self.messages_received,
            "last_message_age": time.time() - self.last_message_time if self.last_message_time else None,
            "subscribed_markets": subscribed_markets,
            "cached_orderbooks": cached_orderbooks,
        }


class UserWebSocket:
    """Authenticated WebSocket client for real-time order status updates.

    Provides real-time notifications for:
    - MATCHED: Order filled
    - MINED: Transaction submitted to chain
    - CONFIRMED: Transaction confirmed
    - FAILED: Order failed
    - RETRYING: Order being retried

    Requires API credentials from py-clob-client.
    """

    USER_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        on_order_update: Callable[[dict], None] | None = None,
    ):
        """Initialize authenticated User WebSocket.

        Args:
            api_key: API key from py-clob-client credentials
            api_secret: API secret
            api_passphrase: API passphrase
            on_order_update: Callback for order status updates
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._on_order_update = on_order_update

        self._ws = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._authenticated = threading.Event()
        self._lock = threading.Lock()

        # Track pending orders for status updates
        self._pending_orders: dict[str, dict] = {}  # order_id -> order info

        # Statistics
        self.reconnect_count = 0
        self.last_message_time = 0.0
        self.messages_received = 0
        self.orders_tracked = 0

    def start(self):
        """Start WebSocket connection in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Wait for authentication (with timeout)
        timeout = 10.0
        start = time.time()
        while not self._authenticated.is_set() and time.time() - start < timeout:
            time.sleep(0.1)

        if not self._authenticated.is_set():
            print("[user-ws] Warning: Authentication timeout")

    def stop(self):
        """Stop WebSocket connection gracefully."""
        self._running = False

        if self._loop and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._graceful_shutdown(), self._loop)
                time.sleep(0.5)
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                # Event loop already closed, ignore
                pass

        if self._thread:
            self._thread.join(timeout=2.0)

    async def _graceful_shutdown(self):
        """Gracefully close WebSocket and cancel tasks."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

        tasks = [t for t in asyncio.all_tasks(self._loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _run_loop(self):
        """Run asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            print(f"[user-ws] Event loop error: {e}")
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()

    async def _connect_loop(self):
        """Main connection loop with reconnection logic."""
        while self._running:
            try:
                async with _ws_connect(
                    self.USER_WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    print(f"[user-ws] Connected to {self.USER_WS_URL}")

                    # Authenticate
                    await self._authenticate()

                    # Message handling loop
                    async for message in ws:
                        self.last_message_time = time.time()
                        self.messages_received += 1
                        await self._handle_message(message)

            except ConnectionClosed as e:
                print(f"[user-ws] Connection closed: {e}")
                self._connected.clear()
                self._authenticated.clear()
            except Exception as e:
                print(f"[user-ws] Connection error: {e}")
                self._connected.clear()
                self._authenticated.clear()

            if self._running:
                self.reconnect_count += 1
                wait_time = min(30, 2 ** min(self.reconnect_count, 5))
                print(f"[user-ws] Reconnecting in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def _authenticate(self):
        """Send authentication message."""
        if not self._ws:
            return

        # Create authentication message
        auth_msg = {
            "type": "subscribe",
            "channel": "user",
            "auth": {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "passphrase": self._api_passphrase,
            },
            "markets": [],  # Subscribe to all markets for our orders
        }

        await self._ws.send(json.dumps(auth_msg))
        print("[user-ws] Authentication message sent")

    async def _handle_message(self, raw: str | bytes):
        """Handle incoming WebSocket message."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", data.get("event_type", ""))

        # Handle authentication response
        if msg_type == "subscribed" or msg_type == "authenticated":
            self._authenticated.set()
            print("[user-ws] Authenticated successfully")
            return

        if msg_type == "error":
            error = data.get("message", data.get("error", "Unknown error"))
            print(f"[user-ws] Error: {error}")
            return

        # Handle order status updates
        if msg_type in ("order", "trade", "order_update"):
            await self._handle_order_update(data)

    async def _handle_order_update(self, data: dict):
        """Handle order status update."""
        order_id = data.get("order_id", data.get("orderId", data.get("id", "")))
        status = data.get("status", data.get("order_status", ""))
        event = data.get("event", data.get("event_type", ""))

        update = {
            "order_id": order_id,
            "status": status,
            "event": event,
            "timestamp": time.time(),
            "data": data,
        }

        # Map events to statuses
        if event == "MATCHED" or status == "MATCHED":
            update["status"] = "filled"
        elif event == "MINED" or status == "MINED":
            update["status"] = "mined"
        elif event == "CONFIRMED" or status == "CONFIRMED":
            update["status"] = "confirmed"
        elif event == "FAILED" or status == "FAILED":
            update["status"] = "failed"
        elif event == "RETRYING" or status == "RETRYING":
            update["status"] = "retrying"
        elif event == "CANCELED" or status == "CANCELED":
            update["status"] = "cancelled"

        # Update pending order if tracked
        with self._lock:
            if order_id in self._pending_orders:
                self._pending_orders[order_id].update(update)

        # Call callback
        if self._on_order_update:
            try:
                self._on_order_update(update)
            except Exception as e:
                print(f"[user-ws] Callback error: {e}")

    def track_order(self, order_id: str, order_info: dict | None = None):
        """Start tracking an order for status updates.

        Args:
            order_id: Order ID to track
            order_info: Optional additional order info
        """
        with self._lock:
            self._pending_orders[order_id] = {
                "order_id": order_id,
                "status": "pending",
                "tracked_at": time.time(),
                **(order_info or {}),
            }
            self.orders_tracked += 1

    def get_order_status(self, order_id: str) -> dict | None:
        """Get current status of a tracked order."""
        with self._lock:
            return self._pending_orders.get(order_id)

    def untrack_order(self, order_id: str):
        """Stop tracking an order."""
        with self._lock:
            self._pending_orders.pop(order_id, None)

    def is_connected(self) -> bool:
        """Check if WebSocket is connected and authenticated."""
        return self._connected.is_set() and self._authenticated.is_set()

    @property
    def stats(self) -> dict:
        """Get connection statistics."""
        return {
            "connected": self._connected.is_set(),
            "authenticated": self._authenticated.is_set(),
            "reconnect_count": self.reconnect_count,
            "messages_received": self.messages_received,
            "orders_tracked": self.orders_tracked,
            "pending_orders": len(self._pending_orders),
            "last_message_age": time.time() - self.last_message_time if self.last_message_time else None,
        }


class MarketDataCache:
    """High-level cache for BTC 5-min market data.

    Combines WebSocket feeds with REST API fallback.
    """

    def __init__(self, use_websocket: bool = True):
        self._rest_client = PolymarketClient()
        self._ws: PolymarketWebSocket | None = None
        self._use_websocket = use_websocket

        # Cache token IDs for BTC 5-min markets
        self._token_cache: dict[int, tuple[str, str]] = {}  # timestamp -> (up_token, down_token)
        self._condition_cache: dict[int, str] = {}  # timestamp -> condition_id
        self._market_cache: dict[int, dict] = {}  # timestamp -> market data
        self._cache_ttl = 60  # seconds

        # Trade callbacks
        self._trade_callbacks: list[Callable[[TradeEvent], None]] = []
        self._trade_callbacks_lock = threading.Lock()

        if use_websocket:
            self._ws = PolymarketWebSocket(on_trade=self._handle_trade)

    def start(self):
        """Start data feeds."""
        if self._ws:
            self._ws.start()
            print("[cache] WebSocket started")

    def stop(self):
        """Stop data feeds."""
        if self._ws:
            self._ws.stop()
            print("[cache] WebSocket stopped")

    def _handle_trade(self, trade: TradeEvent):
        """Internal trade handler - dispatches to callbacks."""
        with self._trade_callbacks_lock:
            callbacks = list(self._trade_callbacks)

        for cb in callbacks:
            try:
                cb(trade)
            except Exception as e:
                print(f"[cache] Trade callback error: {e}")

    def on_trade(self, callback: Callable[[TradeEvent], None]):
        """Register a trade callback."""
        with self._trade_callbacks_lock:
            self._trade_callbacks.append(callback)

    def prefetch_markets(self, timestamps: list[int]):
        """Pre-fetch and cache market data for given timestamps.

        Call this at startup to warm the cache with upcoming markets.
        """
        for ts in timestamps:
            self._fetch_and_cache_market(ts)

    def _fetch_and_cache_market(self, timestamp: int) -> bool:
        """Fetch market data and cache token IDs."""
        if timestamp in self._token_cache:
            return True

        market = self._rest_client.get_market(timestamp)
        if not market:
            return False

        # Cache token IDs
        if market.up_token_id and market.down_token_id:
            self._token_cache[timestamp] = (market.up_token_id, market.down_token_id)

        # Cache market data
        self._market_cache[timestamp] = {
            "up_token_id": market.up_token_id,
            "down_token_id": market.down_token_id,
            "fetched_at": time.time(),
        }

        # Subscribe to WebSocket if available
        if self._ws and self._ws.is_connected():
            # Use slug as condition_id for BTC markets
            token_ids = [t for t in [market.up_token_id, market.down_token_id] if t is not None]
            self._ws.subscribe_market(market.slug, token_ids or None)

        return True

    def get_token_ids(self, timestamp: int) -> tuple[str, str] | None:
        """Get cached token IDs for a market timestamp.

        Returns: (up_token_id, down_token_id) or None
        """
        if timestamp in self._token_cache:
            return self._token_cache[timestamp]

        # Try to fetch
        if self._fetch_and_cache_market(timestamp):
            return self._token_cache.get(timestamp)
        return None

    def get_orderbook(self, token_id: str) -> dict:
        """Get orderbook - from WebSocket cache or REST fallback."""
        # Try WebSocket cache first
        if self._ws and self._ws.is_connected():
            book = self._ws.get_orderbook(token_id)
            if book and book.timestamp > time.time() - 5:  # Max 5s stale
                return {
                    "bids": [{"price": str(l.price), "size": str(l.size)} for l in book.bids],
                    "asks": [{"price": str(l.price), "size": str(l.size)} for l in book.asks],
                    "source": "websocket",
                    "age_ms": int((time.time() - book.timestamp) * 1000),
                }

        # Fallback to REST
        book = self._rest_client.get_orderbook(token_id)
        if book:
            book["source"] = "rest"
        return book

    def get_execution_price(
        self, token_id: str, side: str, amount_usd: float, copy_delay_ms: int = 0
    ) -> tuple[float, float, float, float, float, dict | None]:
        """Get execution price - from WebSocket cache or REST fallback.

        Returns: (exec_price, spread, slippage_pct, fill_pct, delay_impact_pct, delay_breakdown)
        """
        # Try WebSocket cache first
        if self._ws and self._ws.is_connected():
            book = self._ws.get_orderbook(token_id)
            if book and book.timestamp > time.time() - 2:  # Max 2s stale for execution
                return self._ws.get_execution_price(token_id, side, amount_usd, copy_delay_ms)

        # Fallback to REST
        return self._rest_client.get_execution_price(token_id, side, amount_usd, copy_delay_ms)

    def get_mid(self, token_id: str) -> float | None:
        """Get midpoint price - from WebSocket cache or REST fallback."""
        # Try WebSocket cache first
        if self._ws and self._ws.is_connected():
            mid = self._ws.get_mid(token_id)
            if mid is not None:
                return mid

        # Fallback to REST
        return self._rest_client.get_midpoint(token_id)

    @property
    def ws_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws.is_connected() if self._ws else False

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "cached_markets": len(self._token_cache),
            "use_websocket": self._use_websocket,
        }
        if self._ws:
            stats["websocket"] = self._ws.stats
        return stats
