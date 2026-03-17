"""Polymarket CLOB order-book imbalance strategy.

LIVE ONLY — reads real-time order book state from CachedOrderBook (via
PolymarketWebSocket). There is no historical Polymarket order book data,
so this strategy cannot be backtested.

Signal logic:
    1. Depth imbalance: compare total YES depth vs NO implied depth within
       a configurable price band around mid. Imbalance ratio > buy_threshold
       → signal=+1 (YES side has more depth, market implies it's underpriced).
       Ratio < 1/buy_threshold → signal=-1.

    2. Spread gate: if bid-ask spread exceeds max_spread_cents, abstain
       (signal=0). Wide spreads indicate low liquidity and poor fill quality.

    3. Staleness gate: if the order book hasn't been updated within
       max_stale_seconds, abstain.

Integration note:
    This strategy operates on a tick/event basis rather than closed candles.
    The `evaluate_book()` method returns a signal directly from a CachedOrderBook.
    The `evaluate()` method is provided for interface compatibility but always
    returns 0 signals (use evaluate_book() in live loops instead).

    See packages/executor/src/polymarket_algo/executor/ws.py for CachedOrderBook.
"""

from __future__ import annotations

import time

import pandas as pd


class CLOBImbalanceStrategy:
    """Live-only strategy: signal from Polymarket order book depth imbalance.

    Reads YES/NO depth from a live CachedOrderBook snapshot. Cannot be
    backtested — no historical Polymarket CLOB data exists.
    """

    name = "clob_imbalance"
    description = "Live-only: Polymarket CLOB YES/NO depth imbalance signal"
    timeframe = "tick"  # not candle-based

    @property
    def default_params(self) -> dict:
        return {
            "depth_band": 0.05,  # look at bids within 5% of mid
            "buy_threshold": 1.5,  # YES/NO ratio above which signal=+1
            "max_spread": 0.03,  # abstain if spread > 3 cents
            "max_stale_seconds": 10,  # abstain if book is older than this
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        # param_grid is meaningless for live-only strategies, but keep the
        # interface consistent so tooling doesn't break.
        return {
            "depth_band": [0.03, 0.05, 0.10],
            "buy_threshold": [1.3, 1.5, 2.0],
            "max_spread": [0.02, 0.03, 0.05],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Candle-based evaluate — always returns 0 (live use evaluate_book()).

        Provided for interface compatibility with the Strategy protocol.
        In a live loop, call evaluate_book(yes_book, no_book, **params) instead.
        """
        return pd.DataFrame(
            {"signal": 0, "size": 0.0},
            index=candles.index,
        )

    def evaluate_book(
        self,
        yes_book: object,
        no_book: object | None = None,
        **params,
    ) -> int:
        """Evaluate CLOB imbalance from live order book snapshots.

        Args:
            yes_book: CachedOrderBook for the YES token.
            no_book:  CachedOrderBook for the NO token (optional).
                      If provided, compare YES bid depth vs NO bid depth.
                      If None, compare YES bid depth vs YES ask depth.
            **params: depth_band, buy_threshold, max_spread, max_stale_seconds

        Returns:
            1  — YES side deeper (buy YES signal)
            -1 — NO side deeper (sell YES / buy NO signal)
            0  — abstain (spread too wide, book stale, or ratio neutral)
        """
        depth_band = float(params.get("depth_band", self.default_params["depth_band"]))
        buy_threshold = float(params.get("buy_threshold", self.default_params["buy_threshold"]))
        max_spread = float(params.get("max_spread", self.default_params["max_spread"]))
        max_stale = float(params.get("max_stale_seconds", self.default_params["max_stale_seconds"]))

        if yes_book is None:
            return 0

        # Staleness gate
        book_age = time.time() - getattr(yes_book, "timestamp", 0.0)
        if book_age > max_stale:
            return 0

        best_bid = getattr(yes_book, "best_bid", 0.0)
        best_ask = getattr(yes_book, "best_ask", 0.0)
        mid = getattr(yes_book, "mid", 0.5)

        if best_bid <= 0 or best_ask <= 0:
            return 0

        # Spread gate
        spread = best_ask - best_bid
        if spread > max_spread:
            return 0

        # Depth calculation: sum bid sizes within depth_band of mid
        yes_bids = getattr(yes_book, "bids", [])
        yes_bid_depth = sum(level.price * level.size for level in yes_bids if level.price >= mid * (1 - depth_band))

        if no_book is not None:
            # Compare YES bid depth vs NO bid depth
            no_bids = getattr(no_book, "bids", [])
            no_mid = getattr(no_book, "mid", 0.5)
            no_bid_depth = sum(
                level.price * level.size for level in no_bids if level.price >= no_mid * (1 - depth_band)
            )
            if no_bid_depth <= 0:
                return 0
            ratio = yes_bid_depth / no_bid_depth
        else:
            # Compare YES bid depth vs YES ask depth (within band)
            yes_asks = getattr(yes_book, "asks", [])
            yes_ask_depth = sum(level.price * level.size for level in yes_asks if level.price <= mid * (1 + depth_band))
            if yes_ask_depth <= 0:
                return 0
            ratio = yes_bid_depth / yes_ask_depth

        if ratio > buy_threshold:
            return 1
        if ratio < 1.0 / buy_threshold:
            return -1
        return 0
