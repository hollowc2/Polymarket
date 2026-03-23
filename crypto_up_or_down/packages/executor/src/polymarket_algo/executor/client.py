"""Polymarket API client for reading market data and placing trades."""

import json
import math
import time
from dataclasses import dataclass, field

import requests
from polymarket_algo.core.config import Config
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class DelayImpactModel:
    """Non-linear, liquidity-aware model for copy delay price impact.

    Calculates the expected price impact from copying a trade with delay.
    Uses: impact = sqrt(delay) * base_coef * liquidity_factor * volatility_factor
    """

    base_coef: float = field(default_factory=lambda: Config.DELAY_MODEL_BASE_COEF)
    max_impact: float = field(default_factory=lambda: Config.DELAY_MODEL_MAX_IMPACT)
    baseline_spread: float = field(default_factory=lambda: Config.DELAY_MODEL_BASELINE_SPREAD)

    def calculate_impact(
        self,
        delay_ms: int,
        order_size: float = 0.0,
        depth_at_best: float = 0.0,
        spread: float = 0.0,
        side: str = "BUY",
    ) -> tuple[float, dict]:
        """Calculate delay impact percentage.

        Args:
            delay_ms: Milliseconds since the original trade
            order_size: Our order size in USD
            depth_at_best: Available liquidity at best price level
            spread: Current bid-ask spread
            side: "BUY" or "SELL"

        Returns:
            Tuple of (impact_pct, breakdown_dict)
            - impact_pct: Expected price impact as percentage (e.g., 1.5 = 1.5%)
            - breakdown_dict: Detailed calculation breakdown for logging
        """
        if delay_ms <= 0:
            return 0.0, {"delay_ms": 0, "impact_pct": 0.0}

        delay_seconds = delay_ms / 1000.0

        # Base impact: sqrt decay - faster initial impact, slower growth over time
        # sqrt(1s) * 0.8 = 0.8%, sqrt(4s) * 0.8 = 1.6%, sqrt(9s) * 0.8 = 2.4%
        base_impact = math.sqrt(delay_seconds) * self.base_coef

        # Liquidity factor: larger orders relative to available depth = more impact
        # If depth_at_best is 0 or unknown, assume neutral factor of 1.0
        if depth_at_best > 0 and order_size > 0:
            # Ratio of order to 50% of available depth
            # > 1.0 means we're taking more than half the best level
            liq_ratio = order_size / (depth_at_best * 0.5)
            liq_factor = min(2.0, max(0.5, liq_ratio))
        else:
            liq_factor = 1.0

        # Volatility factor: wider spread = more volatile = more impact
        # Baseline spread of 2% (0.02) = neutral factor of 1.0
        if spread > 0 and self.baseline_spread > 0:
            vol_ratio = spread / self.baseline_spread
            vol_factor = min(2.0, max(0.5, vol_ratio))
        else:
            vol_factor = 1.0

        # Final impact
        final_impact = base_impact * liq_factor * vol_factor
        final_impact = min(self.max_impact, final_impact)

        breakdown = {
            "delay_ms": delay_ms,
            "delay_seconds": round(delay_seconds, 2),
            "base_impact": round(base_impact, 4),
            "liquidity_factor": round(liq_factor, 2),
            "volatility_factor": round(vol_factor, 2),
            "final_impact_pct": round(final_impact, 4),
            "order_size": round(order_size, 2),
            "depth_at_best": round(depth_at_best, 2),
            "spread": round(spread, 4),
        }

        return final_impact, breakdown


@dataclass
class Market:
    """A single BTC 5-min up/down market."""

    timestamp: int
    slug: str
    title: str
    closed: bool
    outcome: str | None  # "up", "down", or None if not resolved
    up_token_id: str | None
    down_token_id: str | None
    up_price: float
    down_price: float
    volume: float
    accepting_orders: bool
    taker_fee_bps: int = 1000  # Default 10% base fee
    resolved: bool = False  # True when umaResolutionStatus == "resolved"
    condition_id: str | None = None  # Gnosis CTF conditionId (bytes32 hex) for on-chain redemption
    neg_risk: bool = False  # True if market uses the NegRisk CTF contract


class PolymarketClient:
    """Read-only client for Polymarket APIs (no auth needed).

    Features:
    - Connection pooling for better performance
    - Configurable timeouts and retries
    - Token ID caching for BTC 5-min markets
    """

    def __init__(self, timeout: float | None = None, use_cache: bool = True, asset: str = "btc"):
        self.gamma = Config.GAMMA_API
        self.clob = Config.CLOB_API
        self.timeout = timeout or Config.REST_TIMEOUT
        self.asset = asset.lower()

        # Create session with connection pooling
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=Config.REST_RETRIES,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=retry_strategy,
        )
        self.session.mount("https://", adapter)

        self.session.headers.update(
            {
                "User-Agent": "PolymarketBot/2.0",
                "Accept": "application/json",
                "Connection": "keep-alive",
            }
        )

        # Token ID cache for 5-min markets: timestamp -> (up_token, down_token)
        self._token_cache: dict[int, tuple[str | None, str | None]] = {}
        self._market_cache: dict[int, Market] = {}
        self._cache_ttl = 300  # 5 minutes
        self._use_cache = use_cache

    def get_market(self, timestamp: int, use_cache: bool = True) -> Market | None:
        """Fetch a BTC 5-min market by its timestamp.

        Args:
            timestamp: Unix timestamp of the market
            use_cache: Whether to use cached market data (for token IDs)
        """
        # Check cache first (for recently fetched markets)
        if use_cache and self._use_cache and timestamp in self._market_cache:
            cached = self._market_cache[timestamp]
            # Only return cached if:
            # 1. Market is fully resolved (outcome known) - state is final
            # 2. OR market is still well within its window (prices stable)
            now = int(time.time())
            market_end = timestamp + 300  # 5-min window ends 300s after start

            if cached.closed and cached.outcome:
                # Resolved markets are final - safe to cache forever
                return cached
            elif now < market_end:
                # Market still in window - cache is reasonably fresh
                return cached
            # Otherwise, market may have closed/resolved - fetch fresh data

        slug = f"{self.asset}-updown-5m-{timestamp}"
        try:
            resp = self.session.get(f"{self.gamma}/events", params={"slug": slug}, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return None

            event = data[0]
            markets = event.get("markets", [])
            if not markets:
                return None

            m = markets[0]
            # Parse token IDs
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            up_token = token_ids[0] if len(token_ids) > 0 else None
            down_token = token_ids[1] if len(token_ids) > 1 else None

            # Cache token IDs (these never change)
            self._token_cache[timestamp] = (up_token, down_token)

            # Parse prices
            prices = json.loads(m.get("outcomePrices", "[0.5, 0.5]"))
            up_price = float(prices[0]) if prices else 0.5
            down_price = float(prices[1]) if len(prices) > 1 else 0.5

            # Determine outcome if resolved
            # A market is truly resolved when:
            # 1. closed=true AND
            # 2. umaResolutionStatus="resolved" (or outcomePrices shows 1.0/0.0)
            outcome = None
            is_closed = m.get("closed", False)
            uma_status = m.get("umaResolutionStatus", "")
            is_resolved = uma_status == "resolved"

            if is_closed and (is_resolved or up_price > 0.99 or down_price > 0.99):
                # Use threshold comparison to handle float precision
                if up_price > 0.99:
                    outcome = "up"
                elif down_price > 0.99:
                    outcome = "down"

            # Extract fee rate from market data (already in Gamma response)
            taker_fee_bps = m.get("takerBaseFee")
            if taker_fee_bps is None:
                taker_fee_bps = 1000
                # Only log once per market
                if timestamp not in self._token_cache:
                    print(f"[polymarket] No takerBaseFee in response for {slug}, using default {taker_fee_bps} bps")
            else:
                taker_fee_bps = int(taker_fee_bps)

            market = Market(
                timestamp=timestamp,
                slug=slug,
                title=event.get("title", ""),
                closed=event.get("closed", False) or m.get("closed", False),
                outcome=outcome,
                up_token_id=up_token,
                down_token_id=down_token,
                up_price=up_price,
                down_price=down_price,
                volume=event.get("volume", 0),
                accepting_orders=m.get("acceptingOrders", False),
                taker_fee_bps=taker_fee_bps,
                resolved=is_resolved,
                condition_id=m.get("conditionId"),
                neg_risk=bool(m.get("negRisk", False)),
            )

            # Cache market
            if self._use_cache:
                self._market_cache[timestamp] = market

            return market
        except requests.exceptions.Timeout:
            # Don't spam logs for timeouts
            return None
        except Exception as e:
            print(f"[polymarket] Error fetching {slug}: {e}")
            return None

    def get_token_ids(self, timestamp: int) -> tuple[str | None, str | None]:
        """Get cached token IDs for a market, fetching if needed.

        Returns: (up_token_id, down_token_id)
        """
        if timestamp in self._token_cache:
            return self._token_cache[timestamp]

        # Fetch market to populate cache
        market = self.get_market(timestamp)
        if market:
            return (market.up_token_id, market.down_token_id)
        return (None, None)

    def prefetch_markets(self, timestamps: list[int]) -> int:
        """Pre-fetch and cache multiple markets.

        Returns number of successfully fetched markets.
        """
        success = 0
        for ts in timestamps:
            if self.get_market(ts) is not None:
                success += 1
        return success

    def get_upcoming_market_timestamps(self, count: int = 5) -> list[int]:
        """Get timestamps of upcoming BTC 5-min windows.

        Useful for pre-fetching market data.
        """
        now = int(time.time())
        current_window = (now // 300) * 300
        return [current_window + (i * 300) for i in range(count)]

    def get_recent_outcomes(self, count: int = 10) -> list[str]:
        """Get the last N resolved market outcomes (oldest first)."""
        now = int(time.time())
        current_window = (now // 300) * 300
        outcomes: list[str] = []

        # Walk backwards from the most recent completed window
        ts = current_window - 300  # previous window (should be resolved or resolving)
        attempts = 0
        max_attempts = count + 10  # some buffer for missing markets

        while len(outcomes) < count and attempts < max_attempts:
            market = self.get_market(ts)
            if market and market.closed and market.outcome:
                outcomes.append(market.outcome)
            ts -= 300
            attempts += 1
            time.sleep(0.05)

        # Reverse so oldest is first
        outcomes.reverse()
        return outcomes

    def get_next_market_timestamp(self) -> int:
        """Get the timestamp of the next upcoming 5-min window."""
        now = int(time.time())
        current_window = (now // 300) * 300
        # If we're in the first half of the window, current might still be tradeable
        # But for streak strategy, we want the NEXT unresolved one
        next_window = current_window + 300
        if now - current_window < 60:
            # Window just started, current is still open
            return current_window
        return next_window

    def get_orderbook(self, token_id: str) -> dict:
        """Get order book for a token."""
        try:
            resp = self.session.get(f"{self.clob}/book", params={"token_id": token_id}, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            # Silent timeout - caller can use fallback
            return {}
        except Exception as e:
            print(f"[polymarket] Error fetching orderbook: {e}")
            return {}

    def get_orderbooks(self, token_ids: list[str]) -> dict[str, dict]:
        """Get multiple order books in batch (if supported).

        Falls back to individual requests if batch not available.
        """
        # Try batch endpoint first
        try:
            resp = self.session.get(
                f"{self.clob}/books",
                params={"token_ids": ",".join(token_ids)},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # Fallback to individual requests
        results = {}
        for tid in token_ids:
            book = self.get_orderbook(tid)
            if book:
                results[tid] = book
        return results

    def get_midpoint(self, token_id: str) -> float | None:
        """Get midpoint price for a token.

        Faster than get_orderbook for just getting the mid price.
        Returns None if the price cannot be determined.
        """
        try:
            resp = self.session.get(f"{self.clob}/midpoint", params={"token_id": token_id}, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            mid = data.get("mid")
            return float(mid) if mid is not None else None
        except requests.exceptions.Timeout:
            return None
        except Exception as e:
            print(f"[polymarket] Error fetching midpoint: {e}")
            return None

    def get_price(self, token_id: str, side: str = "BUY") -> float | None:
        """Get best price for a token (fastest endpoint).

        Args:
            token_id: Token to get price for
            side: "BUY" returns best ask, "SELL" returns best bid

        Returns None if the price cannot be determined.
        """
        try:
            resp = self.session.get(
                f"{self.clob}/price",
                params={"token_id": token_id, "side": side},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            price = data.get("price")
            return float(price) if price is not None else None
        except Exception:
            return None

    def get_spread(self, token_id: str) -> tuple[float, float] | None:
        """Get bid-ask spread for a token.

        Returns: (best_bid, best_ask) or None
        """
        try:
            resp = self.session.get(
                f"{self.clob}/spread",
                params={"token_id": token_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return (float(data.get("bid", 0)), float(data.get("ask", 0)))
        except Exception:
            return None

    def get_fee_rate(self, token_id: str) -> int:
        """Get fee rate in basis points for a token.

        Returns base_fee (e.g., 1000 = 10% base rate).
        Actual fee = price * (1 - price) * base_fee / 10000
        """
        DEFAULT_FEE_BPS = 1000  # Fallback: 10% base rate (typical Polymarket fee)
        try:
            resp = self.session.get(f"{self.clob}/fee-rate", params={"token_id": token_id}, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("base_fee", DEFAULT_FEE_BPS))
        except requests.exceptions.Timeout:
            return DEFAULT_FEE_BPS
        except Exception as e:
            print(f"[polymarket] Error fetching fee rate: {e}, using default {DEFAULT_FEE_BPS} bps")
            return DEFAULT_FEE_BPS

    @staticmethod
    def calculate_fee(price: float, base_fee_bps: int) -> float:
        """Calculate actual fee percentage from price and base fee.

        Fee formula: fee = price * (1 - price) * base_fee / 10000
        At 50¢ with base_fee=1000: 0.50 * 0.50 * 0.10 = 2.5%
        """
        if base_fee_bps == 0:
            return 0.0
        return price * (1 - price) * base_fee_bps / 10000

    # ── Limit order helpers (require py-clob-client with auth) ────────────────

    def place_limit_order(self, token_id: str, price: float, shares: float) -> str | None:
        """Post a GTC limit buy order on the live CLOB.

        Args:
            token_id: Token to buy
            price: Limit price (0–1)
            shares: Number of shares to buy

        Returns:
            order_id string if successful, None on failure.

        Note: Requires py-clob-client ClobClient with API credentials.
              This method is a thin wrapper; trading logic lives in LiveTrader.
        """
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            order_args = OrderArgs(token_id=token_id, price=price, size=shares, side=BUY)
            signed = self._clob_client.create_order(order_args)  # type: ignore[unresolved-attribute]
            resp = self._clob_client.post_order(signed, OrderType.GTC)  # type: ignore[unresolved-attribute, invalid-argument-type]
            resp_dict: dict = resp if isinstance(resp, dict) else {}
            return resp_dict.get("orderID", resp_dict.get("id"))
        except AttributeError:
            print("[polymarket] place_limit_order requires an authenticated ClobClient (self._clob_client not set)")
            return None
        except Exception as e:
            print(f"[polymarket] place_limit_order failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open GTC order.

        Returns True on success, False on failure.
        Requires authenticated ClobClient.
        """
        try:
            self._clob_client.cancel(order_id)  # type: ignore[unresolved-attribute]
            return True
        except AttributeError:
            print("[polymarket] cancel_order requires an authenticated ClobClient (self._clob_client not set)")
            return False
        except Exception as e:
            print(f"[polymarket] cancel_order({order_id}) failed: {e}")
            return False

    def get_order_status(self, order_id: str) -> dict:
        """Return fill status for an open order.

        Returns dict with keys: filled (bool), fill_price (float|None), filled_size (float).
        Requires authenticated ClobClient.
        """
        try:
            order = self._clob_client.get_order(order_id)  # type: ignore[unresolved-attribute]
            if not isinstance(order, dict):
                return {"filled": False, "fill_price": None, "filled_size": 0.0}
            status = order.get("status", "").upper()
            if status in ("FILLED", "MATCHED"):
                return {
                    "filled": True,
                    "fill_price": float(order.get("price", 0)),
                    "filled_size": float(order.get("size_matched", order.get("size", 0))),
                }
            return {"filled": False, "fill_price": None, "filled_size": 0.0}
        except AttributeError:
            print("[polymarket] get_order_status requires an authenticated ClobClient (self._clob_client not set)")
            return {"filled": False, "fill_price": None, "filled_size": 0.0}
        except Exception as e:
            print(f"[polymarket] get_order_status({order_id}) failed: {e}")
            return {"filled": False, "fill_price": None, "filled_size": 0.0}

    def get_execution_price(
        self, token_id: str, side: str, amount_usd: float, copy_delay_ms: int = 0
    ) -> tuple[float, float, float, float, float, dict | None]:
        """Calculate execution price with slippage for a given order size.

        Args:
            token_id: The token to trade
            side: "BUY" or "SELL"
            amount_usd: Order size in USD
            copy_delay_ms: Milliseconds since the original trade (for copytrade)

        Returns:
            tuple of (execution_price, spread, slippage_pct, fill_pct, delay_impact_pct, delay_breakdown)
            - execution_price: The price you'll actually get
            - spread: Bid-ask spread
            - slippage_pct: Slippage from walking the book
            - fill_pct: Percentage of order that can be filled (100 = full fill)
            - delay_impact_pct: Additional price impact from copy delay
            - delay_breakdown: Detailed breakdown of delay model calculation (or None)
        """
        book = self.get_orderbook(token_id)
        if not book:
            return (0.0, 0.0, 0.0, 0.0, 0.0, None)

        # Get best bid/ask for spread calculation
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if not bids or not asks:
            return (0.0, 0.0, 0.0, 0.0, 0.0, None)

        # Sort: asks ascending (lowest first), bids descending (highest first)
        asks_sorted = sorted(asks, key=lambda x: float(x["price"]))
        bids_sorted = sorted(bids, key=lambda x: float(x["price"]), reverse=True)

        best_ask = float(asks_sorted[0]["price"])
        best_bid = float(bids_sorted[0]["price"])
        spread = best_ask - best_bid

        # Calculate depth at best price level
        if side == "BUY":
            # Depth at best ask
            best_level = asks_sorted[0]
            depth_at_best = float(best_level["price"]) * float(best_level["size"])
            levels = asks_sorted
        else:
            # Depth at best bid
            best_level = bids_sorted[0]
            depth_at_best = float(best_level["price"]) * float(best_level["size"])
            levels = bids_sorted

        remaining_usd = amount_usd
        total_shares = 0.0
        total_cost = 0.0

        for level in levels:
            price = float(level["price"])
            size = float(level["size"])
            level_value = price * size  # USD value at this level

            if remaining_usd <= 0:
                break

            if level_value >= remaining_usd:
                # This level can fill the rest
                shares_to_take = remaining_usd / price
                total_shares += shares_to_take
                total_cost += remaining_usd
                remaining_usd = 0
            else:
                # Take entire level
                total_shares += size
                total_cost += level_value
                remaining_usd -= level_value

        # Calculate fill percentage
        filled_amount = amount_usd - remaining_usd
        fill_pct = (filled_amount / amount_usd * 100) if amount_usd > 0 else 100.0

        if total_shares == 0:
            return (0.0, spread, 0.0, 0.0, 0.0, None)

        execution_price = total_cost / total_shares

        # Calculate slippage vs best price
        if side == "BUY":
            slippage_pct = (execution_price - best_ask) / best_ask * 100 if best_ask > 0 else 0
        else:
            slippage_pct = (best_bid - execution_price) / best_bid * 100 if best_bid > 0 else 0

        # Calculate copy delay price impact using the improved model
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

            # Apply delay impact to execution price
            if side == "BUY":
                execution_price *= 1 + delay_impact_pct / 100
            else:
                execution_price *= 1 - delay_impact_pct / 100

            # Cap execution price at reasonable bounds
            execution_price = max(0.01, min(0.99, execution_price))

        return (
            execution_price,
            spread,
            max(0, slippage_pct),
            fill_pct,
            delay_impact_pct,
            delay_breakdown,
        )
