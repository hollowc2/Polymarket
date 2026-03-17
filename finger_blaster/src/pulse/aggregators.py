"""Data aggregation and transformation for Pulse."""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Optional

from src.pulse.config import (
    BucketedOrderBook,
    Candle,
    PulseConfig,
    Timeframe,
    Trade,
)

logger = logging.getLogger("Pulse.Aggregators")


class CandleAggregator:
    def __init__(
        self,
        product_id: str = "BTC-USD",
        on_candle: Optional[Callable[[Candle], Coroutine]] = None,
        wall_clock_aligned: bool = True,
        use_prev_close_for_empty: bool = True,
    ):
        self.product_id = product_id
        self.on_candle = on_candle
        self.wall_clock_aligned = wall_clock_aligned
        self.use_prev_close_for_empty = use_prev_close_for_empty

        # Current candle state
        self._current_candle_start: Optional[int] = None
        self._current_ohlcv: Optional[Dict[str, float]] = None
        self._last_close: Optional[float] = None

        # Buffered trades for current candle
        self._trade_buffer: List[Trade] = []

    def _get_candle_start(self, timestamp: float) -> int:
        if self.wall_clock_aligned:
            return int(timestamp // 10) * 10
        else:
            # If not aligned, use first trade time as base
            if self._current_candle_start is None:
                return int(timestamp)
            return self._current_candle_start

    async def add_trade(self, trade: Trade) -> Optional[Candle]:
        candle_start = self._get_candle_start(trade.timestamp)
        completed_candle = None

        # Check if we need to close current candle
        if self._current_candle_start is not None and candle_start > self._current_candle_start:
            # Finalize current candle
            completed_candle = self._finalize_candle()

            # Emit any empty candles for gaps
            await self._emit_empty_candles(
                self._current_candle_start + 10,
                candle_start
            )

            # Start new candle
            self._start_new_candle(candle_start, trade)

            # Emit completed candle
            if completed_candle and self.on_candle:
                await self.on_candle(completed_candle)

            return completed_candle

        # Update or start current candle
        if self._current_candle_start is None:
            self._start_new_candle(candle_start, trade)
        else:
            self._update_candle(trade)

        return None

    def _start_new_candle(self, start: int, trade: Trade):
        self._current_candle_start = start
        self._current_ohlcv = {
            'open': trade.price,
            'high': trade.price,
            'low': trade.price,
            'close': trade.price,
            'volume': trade.size,
        }
        self._trade_buffer = [trade]

    def _update_candle(self, trade: Trade):
        if self._current_ohlcv is None:
            return

        self._current_ohlcv['high'] = max(self._current_ohlcv['high'], trade.price)
        self._current_ohlcv['low'] = min(self._current_ohlcv['low'], trade.price)
        self._current_ohlcv['close'] = trade.price
        self._current_ohlcv['volume'] += trade.size
        self._trade_buffer.append(trade)

    def _finalize_candle(self) -> Optional[Candle]:
        if self._current_candle_start is None or self._current_ohlcv is None:
            return None

        candle = Candle(
            timestamp=self._current_candle_start,
            open=self._current_ohlcv['open'],
            high=self._current_ohlcv['high'],
            low=self._current_ohlcv['low'],
            close=self._current_ohlcv['close'],
            volume=self._current_ohlcv['volume'],
            timeframe=Timeframe.TEN_SEC,
        )

        self._last_close = candle.close
        self._trade_buffer = []

        return candle

    async def _emit_empty_candles(self, start: int, end: int):
        if not self.use_prev_close_for_empty:
            return

        if self._last_close is None:
            return

        current = start
        while current < end:
            empty_candle = Candle(
                timestamp=current,
                open=self._last_close,
                high=self._last_close,
                low=self._last_close,
                close=self._last_close,
                volume=0.0,
                timeframe=Timeframe.TEN_SEC,
            )

            if self.on_candle:
                await self.on_candle(empty_candle)

            current += 10

    async def flush(self) -> Optional[Candle]:
        if self._current_ohlcv is None:
            return None

        candle = self._finalize_candle()
        self._current_candle_start = None
        self._current_ohlcv = None

        if candle and self.on_candle:
            await self.on_candle(candle)

        return candle

    def reset(self):
        self._current_candle_start = None
        self._current_ohlcv = None
        self._last_close = None
        self._trade_buffer = []


class OrderBookBucketer:
    def __init__(
        self,
        bucket_size: float = 100.0,
        product_id: str = "BTC-USD",
        on_update: Optional[Callable[[BucketedOrderBook], Coroutine]] = None,
    ):
        self.bucket_size = bucket_size
        self.product_id = product_id
        self.on_update = on_update

        # Raw order book (price -> size)
        self._raw_bids: Dict[float, float] = {}
        self._raw_asks: Dict[float, float] = {}

        # Bucketed order book
        self._bucketed_bids: Dict[float, float] = {}
        self._bucketed_asks: Dict[float, float] = {}

        # Book state
        self._last_update_time: float = 0.0
        self._snapshot_received: bool = False

    def _bucket_price(self, price: float) -> float:
        return (price // self.bucket_size) * self.bucket_size

    async def process_snapshot(self, data: Dict[str, Any]) -> BucketedOrderBook:
        self._raw_bids.clear()
        self._raw_asks.clear()

        updates = data.get('updates', [])
        for update in updates:
            price = float(update.get('price_level', 0))
            size = float(update.get('new_quantity', 0))
            side = update.get('side', '').lower()

            if price <= 0:
                continue

            if side == 'bid':
                if size > 0:
                    self._raw_bids[price] = size
            elif side in ('ask', 'offer'):
                if size > 0:
                    self._raw_asks[price] = size

        self._snapshot_received = True
        return await self._rebuild_bucketed_book()

    async def process_update(self, data: Dict[str, Any]) -> Optional[BucketedOrderBook]:
        if not self._snapshot_received:
            logger.debug("Ignoring L2 update before snapshot")
            return None

        updates = data.get('updates', [])
        for update in updates:
            price = float(update.get('price_level', 0))
            size = float(update.get('new_quantity', 0))
            side = update.get('side', '').lower()

            if price <= 0:
                continue

            if side == 'bid':
                if size > 0:
                    self._raw_bids[price] = size
                else:
                    self._raw_bids.pop(price, None)
            elif side in ('ask', 'offer'):
                if size > 0:
                    self._raw_asks[price] = size
                else:
                    self._raw_asks.pop(price, None)

        return await self._rebuild_bucketed_book()

    async def _rebuild_bucketed_book(self) -> BucketedOrderBook:
        self._bucketed_bids.clear()
        self._bucketed_asks.clear()

        for price, size in self._raw_bids.items():
            bucket = self._bucket_price(price)
            self._bucketed_bids[bucket] = self._bucketed_bids.get(bucket, 0.0) + size

        for price, size in self._raw_asks.items():
            bucket = self._bucket_price(price)
            self._bucketed_asks[bucket] = self._bucketed_asks.get(bucket, 0.0) + size

        best_bid, best_ask, mid_price, spread = self._compute_book_metrics()
        self._last_update_time = time.time()

        book = BucketedOrderBook(
            bids=dict(self._bucketed_bids),
            asks=dict(self._bucketed_asks),
            mid_price=mid_price,
            spread=spread,
            best_bid=best_bid,
            best_ask=best_ask,
            timestamp=self._last_update_time,
            bucket_size=self.bucket_size,
        )

        if self.on_update:
            await self.on_update(book)

        return book

    def _compute_book_metrics(self) -> tuple:
        best_bid = max(self._raw_bids.keys()) if self._raw_bids else 0.0
        best_ask = min(self._raw_asks.keys()) if self._raw_asks else 0.0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
        spread = (best_ask - best_bid) if best_bid and best_ask else 0.0
        return best_bid, best_ask, mid_price, spread

    def get_current_book(self) -> Optional[BucketedOrderBook]:
        if not self._snapshot_received:
            return None
        best_bid, best_ask, mid_price, spread = self._compute_book_metrics()
        return BucketedOrderBook(
            bids=dict(self._bucketed_bids),
            asks=dict(self._bucketed_asks),
            mid_price=mid_price,
            spread=spread,
            best_bid=best_bid,
            best_ask=best_ask,
            timestamp=self._last_update_time,
            bucket_size=self.bucket_size,
        )

    def get_depth_profile(self, levels: int = 10) -> Dict[str, List[Dict[str, float]]]:
        # Sort bids (highest first)
        sorted_bids = sorted(
            self._bucketed_bids.items(),
            key=lambda x: x[0],
            reverse=True
        )[:levels]

        # Sort asks (lowest first)
        sorted_asks = sorted(
            self._bucketed_asks.items(),
            key=lambda x: x[0]
        )[:levels]

        return {
            'bids': [{'price': p, 'size': s} for p, s in sorted_bids],
            'asks': [{'price': p, 'size': s} for p, s in sorted_asks],
        }

    def reset(self):
        self._raw_bids.clear()
        self._raw_asks.clear()
        self._bucketed_bids.clear()
        self._bucketed_asks.clear()
        self._snapshot_received = False
        self._last_update_time = 0.0


class TimeframeAggregator:
    AGGREGATION_LEVELS = {
        Timeframe.TEN_SEC: (6, Timeframe.ONE_MIN),
        Timeframe.ONE_MIN: (5, Timeframe.FIVE_MIN),
        Timeframe.FIVE_MIN: (3, Timeframe.FIFTEEN_MIN),
        Timeframe.FIFTEEN_MIN: (4, Timeframe.ONE_HOUR),
        Timeframe.ONE_HOUR: (4, Timeframe.FOUR_HOUR),
        Timeframe.FOUR_HOUR: (6, Timeframe.ONE_DAY),
    }

    def __init__(
        self,
        product_id: str = "BTC-USD",
        on_candle: Optional[Callable[[Candle], Coroutine]] = None,
    ):
        self.product_id = product_id
        self.on_candle = on_candle

        # Per-timeframe buffers: timeframe -> list of candles
        self._buffers: Dict[Timeframe, List[Candle]] = {
            tf: [] for tf in self.AGGREGATION_LEVELS.keys()
        }

    async def add_candle(self, candle: Candle) -> List[Candle]:
        completed = []

        # Add to buffer for this timeframe
        self._buffers[candle.timeframe].append(candle)

        # Check if we can aggregate to next timeframe
        if candle.timeframe in self.AGGREGATION_LEVELS:
            count_needed, target_tf = self.AGGREGATION_LEVELS[candle.timeframe]
            buffer = self._buffers[candle.timeframe]

            while len(buffer) >= count_needed:
                # Take candles for aggregation
                source_candles = buffer[:count_needed]

                # Aggregate into target candle
                aggregated = self._aggregate_candles(source_candles, target_tf)
                completed.append(aggregated)

                # Emit via callback
                if self.on_candle:
                    await self.on_candle(aggregated)

                # Remove processed candles from buffer
                buffer[:count_needed] = []

                # Recursively aggregate further if applicable
                if aggregated.timeframe in self.AGGREGATION_LEVELS:
                    further = await self.add_candle(aggregated)
                    completed.extend(further)

        return completed

    def _aggregate_candles(
        self,
        candles: List[Candle],
        target_timeframe: Timeframe
    ) -> Candle:
        if not candles:
            raise ValueError("Cannot aggregate empty candle list")

        # Use first candle's timestamp as base
        start_timestamp = candles[0].timestamp

        # OHLCV aggregation
        open_price = candles[0].open
        high_price = max(c.high for c in candles)
        low_price = min(c.low for c in candles)
        close_price = candles[-1].close
        total_volume = sum(c.volume for c in candles)

        return Candle(
            timestamp=start_timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
            timeframe=target_timeframe,
        )

    def reset(self):
        for buffer in self._buffers.values():
            buffer.clear()
