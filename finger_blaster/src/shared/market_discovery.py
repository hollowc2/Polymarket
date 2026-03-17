"""
Market Discovery Service - Shared module for discovering active 15m BTC markets.

Provides:
- Polling for active markets via PolymarketConnector
- Callback emission on market changes
- Thread-safe access to current market data

This is a thin polling wrapper around PolymarketConnector.get_active_market(),
ensuring a single source of truth for market discovery logic.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.connectors.polymarket import PolymarketConnector

logger = logging.getLogger("FingerBlaster.MarketDiscovery")


class MarketDiscoveryConfig:
    """Configuration for market discovery service."""

    def __init__(
        self,
        series_id: str = "10192",  # BTC Up or Down 15m series
        poll_interval: float = 5.0,  # Seconds between polls
    ):
        self.series_id = series_id
        self.poll_interval = poll_interval


class MarketDiscoveryService:
    """
    Service for discovering and tracking active Polymarket 15m BTC markets.

    Wraps PolymarketConnector.get_active_market() with polling and change detection.
    Thread-safe and designed for use with asyncio.
    """

    def __init__(
        self,
        connector: "PolymarketConnector",
        config: Optional[MarketDiscoveryConfig] = None,
        on_market_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize market discovery service.

        Args:
            connector: PolymarketConnector instance for API calls
            config: Configuration for polling behavior
            on_market_change: Callback when active market changes
        """
        self._connector = connector
        self.config = config or MarketDiscoveryConfig()
        self.on_market_change = on_market_change

        self._current_market: Optional[Dict[str, Any]] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the market discovery polling loop."""
        if self._running:
            logger.debug("MarketDiscoveryService already running")
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("MarketDiscoveryService started")

    async def stop(self) -> None:
        """Stop the market discovery service."""
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        logger.info("MarketDiscoveryService stopped")

    async def get_current_market(self) -> Optional[Dict[str, Any]]:
        """Get the currently active market (thread-safe)."""
        async with self._lock:
            return self._current_market.copy() if self._current_market else None

    async def get_token_map(self) -> Dict[str, str]:
        """Get token map for current market (Up/Down -> token_id)."""
        async with self._lock:
            if self._current_market:
                return self._current_market.get("token_map", {}).copy()
            return {}

    async def fetch_active_market(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the currently active market.

        Delegates to PolymarketConnector.get_active_market() for the actual
        API call and parsing logic.

        Returns:
            Market data dictionary or None if no active market found
        """
        try:
            return await self._connector.get_active_market(self.config.series_id)
        except Exception as e:
            logger.error(f"Error fetching active market: {e}", exc_info=True)
            return None

    async def _poll_loop(self) -> None:
        """Main polling loop for market discovery."""
        logger.info(f"Market discovery poll loop started (interval: {self.config.poll_interval}s)")
        while self._running:
            try:
                logger.info("Polling for active market...")
                market = await self.fetch_active_market()

                if market:
                    # Check if market changed
                    current_id = None
                    new_id = market.get("market_id")

                    async with self._lock:
                        current_id = (
                            self._current_market.get("market_id")
                            if self._current_market
                            else None
                        )

                    logger.info(f"Current market ID: {current_id}, New market ID: {new_id}, New title: {market.get('title', 'Unknown')}")

                    if current_id != new_id:
                        logger.warning(
                            f"MARKET CHANGED DETECTED: {current_id} -> {new_id}"
                        )
                        logger.warning(f"New market title: {market.get('title', 'Unknown')}")

                        async with self._lock:
                            self._current_market = market

                        # Emit callback (OUTSIDE the lock to avoid deadlocks)
                        if self.on_market_change:
                            logger.warning("Calling on_market_change callback")
                            try:
                                if asyncio.iscoroutinefunction(self.on_market_change):
                                    await self.on_market_change(market)
                                else:
                                    self.on_market_change(market)
                                logger.warning("on_market_change callback completed successfully")
                            except Exception as e:
                                logger.error(f"Error in market change callback: {e}", exc_info=True)
                        else:
                            logger.error("WARNING: No on_market_change callback registered!")
                    else:
                        # Update market data even if ID same (strike might resolve)
                        logger.debug(f"Market ID unchanged ({current_id}), updating market data")
                        async with self._lock:
                            self._current_market = market
                else:
                    logger.warning("No active market found in poll")

            except Exception as e:
                logger.error(f"Error in poll loop: {e}", exc_info=True)

            await asyncio.sleep(self.config.poll_interval)
