"""Binance aggTrades fetcher — Cumulative Volume Delta (CVD) per candle.

CVD measures whether buy or sell aggression dominates within each candle.
Divergences between price direction and CVD direction are potential
reversal signals.

Aggressor convention (Binance aggTrades):
    m=True  → market order hit the BID (sell aggressor, bearish)
    m=False → market order hit the ASK (buy aggressor, bullish)

Endpoints (no auth required):
  Historical: GET https://api.binance.com/api/v3/aggTrades
  Live WS:    wss://stream.binance.com:9443/ws/btcusdt@aggTrade

Note: aggTrades can be high volume (many thousands per day for BTC).
Expect slower data fetching compared to OHLCV or funding rate data.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pandas as pd
import requests

AGG_TRADES_URL = "https://api.binance.com/api/v3/aggTrades"
AGG_TRADE_WS_URL = "wss://stream.binance.com:9443/ws/{symbol}@aggTrade"
MAX_LIMIT = 1000


def fetch_agg_trades(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch historical aggregated trades from Binance spot.

    Args:
        symbol: Spot symbol, e.g. "BTCUSDT"
        start_ms: Start timestamp in milliseconds (UTC)
        end_ms: End timestamp in milliseconds (UTC)

    Returns:
        DataFrame with columns:
            timestamp   (UTC datetime)
            price       (float)
            qty         (float, base asset)
            is_buy_agg  (bool, True = buy aggressor hit ask)
        Sorted ascending by timestamp.
    """
    rows: list[dict] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": min(cursor + 3600 * 1000, end_ms),  # 1-hour windows to keep pages manageable
            "limit": MAX_LIMIT,
        }
        resp = requests.get(AGG_TRADES_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            cursor = params["endTime"] + 1
            continue

        rows.extend(data)
        last_ts = int(data[-1]["T"])
        if last_ts <= cursor:
            cursor = params["endTime"] + 1
        else:
            cursor = last_ts + 1
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "price", "qty", "is_buy_agg"])

    df = pd.DataFrame(rows)
    # Binance aggTrades fields: T=time, p=price, q=qty, m=is_maker (m=True → sell aggressor)
    df = df.rename(columns={"T": "timestamp_ms", "p": "price", "q": "qty", "m": "is_maker"})
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    # m=True means the buyer was the maker → sell aggressor. Invert for buy_agg.
    df["is_buy_agg"] = ~df["is_maker"].astype(bool)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return df[["timestamp", "price", "qty", "is_buy_agg"]]


def compute_cvd_candles(trades_df: pd.DataFrame, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-candle CVD and merge onto candle DataFrame.

    Columns added:
        buy_vol    — total buy-aggressor quantity in this candle
        sell_vol   — total sell-aggressor quantity in this candle
        delta      — buy_vol - sell_vol (signed net flow for the candle)
        cvd        — cumulative sum of delta over all candles (running CVD)

    Args:
        trades_df: Output of fetch_agg_trades().
        candles_df: OHLCV DataFrame indexed by tz-aware UTC open_time.

    Returns:
        candles_df with buy_vol, sell_vol, delta, cvd columns merged in.
        Candles with no trades have 0.0 for all four columns.
    """
    zero_cols = {"buy_vol": 0.0, "sell_vol": 0.0, "delta": 0.0, "cvd": 0.0}

    if trades_df.empty or candles_df.empty:
        out = candles_df.copy()
        for col, val in zero_cols.items():
            out[col] = val
        return out

    # Infer candle period
    if len(candles_df) > 1:
        gaps = candles_df.index.to_series().diff().dropna()
        candle_period = gaps.median()
    else:
        candle_period = pd.Timedelta("5min")

    trades = trades_df.copy()
    trades["open_time"] = trades["timestamp"].dt.floor(candle_period)

    buy_trades = trades[trades["is_buy_agg"]]
    sell_trades = trades[~trades["is_buy_agg"]]

    buy_vol = buy_trades.groupby("open_time")["qty"].sum()
    sell_vol = sell_trades.groupby("open_time")["qty"].sum()

    out = candles_df.copy()
    out["buy_vol"] = buy_vol.reindex(out.index).fillna(0.0).values
    out["sell_vol"] = sell_vol.reindex(out.index).fillna(0.0).values
    out["delta"] = out["buy_vol"] - out["sell_vol"]
    out["cvd"] = out["delta"].cumsum()

    return out


class AggTradeCollector:
    """Live collector: subscribes to Binance aggTrade WebSocket stream.

    Accumulates buy/sell volume per open candle. Call get_current_delta()
    at candle close to get the completed candle's delta, then reset.

    Usage:
        collector = AggTradeCollector("btcusdt")
        collector.start()
        # ... at each candle close:
        delta = collector.get_current_delta()
        collector.reset()
        # ... on shutdown:
        collector.stop()
    """

    def __init__(self, symbol: str = "btcusdt") -> None:
        self.symbol = symbol.lower()
        self._buy_vol: float = 0.0
        self._sell_vol: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start the WebSocket collector in a background daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="agg-trade-collector")
        self._thread.start()

    def stop(self) -> None:
        """Stop the WebSocket collector."""
        self._running = False
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)

    def reset(self) -> None:
        """Reset accumulators at the start of a new candle."""
        with self._lock:
            self._buy_vol = 0.0
            self._sell_vol = 0.0

    def get_current_delta(self) -> float:
        """Return delta (buy_vol - sell_vol) accumulated since last reset."""
        with self._lock:
            return self._buy_vol - self._sell_vol

    def get_current_volumes(self) -> tuple[float, float]:
        """Return (buy_vol, sell_vol) accumulated since last reset."""
        with self._lock:
            return self._buy_vol, self._sell_vol

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as exc:
            if self._running:
                print(f"[agg-trades] Event loop error: {exc}")
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

    async def _connect_loop(self) -> None:
        # Import here to avoid hard dependency at module level for non-live usage
        try:
            import websockets
            from websockets.exceptions import ConnectionClosed
        except ImportError:
            print("[agg-trades] websockets package not installed — live collector unavailable")
            return

        url = AGG_TRADE_WS_URL.format(symbol=self.symbol)
        reconnect_count = 0

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    print(f"[agg-trades] Connected: {url}")
                    reconnect_count = 0
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        qty = float(msg.get("q", 0))
                        is_maker = bool(msg.get("m", False))
                        with self._lock:
                            if is_maker:
                                # buyer is maker → sell aggressor
                                self._sell_vol += qty
                            else:
                                self._buy_vol += qty
            except ConnectionClosed as exc:
                if self._running:
                    print(f"[agg-trades] Disconnected: {exc}")
            except Exception as exc:
                if self._running:
                    print(f"[agg-trades] Error: {exc}")

            if self._running:
                reconnect_count += 1
                wait = min(30, 2 ** min(reconnect_count, 5))
                print(f"[agg-trades] Reconnecting in {wait}s…")
                await asyncio.sleep(wait)


class LiquidationCollector:
    """Live collector: subscribes to Binance futures forceOrder stream.

    Accumulates long and short liquidation USD values per open candle.

    Usage:
        collector = LiquidationCollector("btcusdt")
        collector.start()
        # ... at each candle close:
        long_usd, short_usd = collector.get_current_liqs()
        collector.reset()
        collector.stop()
    """

    FORCE_ORDER_WS_URL = "wss://fstream.binance.com/stream?streams={symbol}@forceOrder"

    def __init__(self, symbol: str = "btcusdt") -> None:
        self.symbol = symbol.lower()
        self._long_liq_usd: float = 0.0  # SELL-side force orders (longs blown)
        self._short_liq_usd: float = 0.0  # BUY-side force orders (shorts blown)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="liq-collector")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)

    def reset(self) -> None:
        with self._lock:
            self._long_liq_usd = 0.0
            self._short_liq_usd = 0.0

    def get_current_liqs(self) -> tuple[float, float]:
        """Return (long_liq_usd, short_liq_usd) accumulated since last reset."""
        with self._lock:
            return self._long_liq_usd, self._short_liq_usd

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as exc:
            if self._running:
                print(f"[liq-collector] Event loop error: {exc}")
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

    async def _connect_loop(self) -> None:
        try:
            import websockets
            from websockets.exceptions import ConnectionClosed
        except ImportError:
            print("[liq-collector] websockets package not installed — live collector unavailable")
            return

        url = self.FORCE_ORDER_WS_URL.format(symbol=self.symbol)
        reconnect_count = 0

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    print(f"[liq-collector] Connected: {url}")
                    reconnect_count = 0
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            wrapper = json.loads(raw)
                            # forceOrder stream format: {"stream":"...", "data":{"e":"forceOrder","o":{...}}}
                            order = wrapper.get("data", {}).get("o", {})
                        except (json.JSONDecodeError, AttributeError):
                            continue

                        side = order.get("S", "")
                        price = float(order.get("ap", order.get("p", 0)))
                        qty = float(order.get("q", 0))
                        usd_val = price * qty

                        with self._lock:
                            if side == "SELL":
                                # SELL side = long positions being liquidated
                                self._long_liq_usd += usd_val
                            elif side == "BUY":
                                # BUY side = short positions being liquidated
                                self._short_liq_usd += usd_val

            except ConnectionClosed as exc:
                if self._running:
                    print(f"[liq-collector] Disconnected: {exc}")
            except Exception as exc:
                if self._running:
                    print(f"[liq-collector] Error: {exc}")

            if self._running:
                reconnect_count += 1
                wait = min(30, 2 ** min(reconnect_count, 5))
                print(f"[liq-collector] Reconnecting in {wait}s…")
                await asyncio.sleep(wait)
