"""Binance futures forced-order (liquidation) fetcher.

Liquidation cascades occur when a large concentration of positions is
force-closed in a short window. They cause sudden, sharp directional
moves followed by a snap-back as liquidity re-enters.

Side convention (Binance):
    BUY  side = short positions being force-closed (short liquidations)
    SELL side = long positions being force-closed (long liquidations)

Endpoint (no auth required):
    GET https://fapi.binance.com/fapi/v1/allForceOrders

Note: the endpoint returns at most 7 days of history per query. For
longer backtests you must make overlapping requests.

⚠ DEPRECATION: As of 2024, Binance deprecated `GET /fapi/v1/allForceOrders`
("The endpoint has been out of maintenance"). fetch_liquidations() returns an
empty DataFrame when the endpoint is unavailable — strategies degrade gracefully
to all-zero liq columns. Live data is still available via WebSocket:
    wss://fstream.binance.com/stream?streams=btcusdt@forceOrder
(see LiquidationCollector in agg_trades.py).

For historical liquidation data, third-party sources are required (CoinGlass,
Glassnode, etc.). Consider storing live WebSocket data locally going forward.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

FAPI_BASE = "https://fapi.binance.com"
FORCE_ORDERS_URL = f"{FAPI_BASE}/fapi/v1/allForceOrders"
MAX_LIMIT = 1000
# Binance retains ~90 days of force-order history. Jump to this lookback
# on the first 400 instead of iterating through every old 7-day window.
LIQ_HISTORY_DAYS = 90


def fetch_liquidations(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch historical forced-liquidation orders from Binance futures.

    Args:
        symbol: Futures symbol, e.g. "BTCUSDT"
        start_ms: Start timestamp in milliseconds (UTC)
        end_ms: End timestamp in milliseconds (UTC)

    Returns:
        DataFrame with columns:
            timestamp   (UTC datetime)
            side        ("BUY" = short liq, "SELL" = long liq)
            price       (float, average fill price)
            qty         (float, base asset quantity)
            usd_value   (float, approx USD notional = price * qty)
        Sorted ascending by timestamp.
    """
    rows: list[dict] = []
    cursor = start_ms
    # Track whether we've already jumped forward to the available-data window.
    _jumped_to_recent = False

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": min(cursor + 7 * 24 * 3600 * 1000, end_ms),  # max 7-day window
            "limit": MAX_LIMIT,
        }
        try:
            resp = requests.get(FORCE_ORDERS_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 400:
                # 400 can mean either the endpoint is deprecated (permanent) or the
                # request falls outside the available history window (temporary).
                # On first 400, try jumping to 90 days ago. If still 400, give up.
                if not _jumped_to_recent:
                    _jumped_to_recent = True
                    cutoff_ms = int(time.time() * 1000) - LIQ_HISTORY_DAYS * 24 * 3600 * 1000
                    if cutoff_ms > cursor:
                        cursor = cutoff_ms
                        continue
                # Still 400 after jump (or endpoint deprecated) — stop gracefully.
                break
            raise
        data = resp.json()

        if not data:
            # No liquidations in this window — advance by the window size
            cursor = params["endTime"] + 1
            continue

        rows.extend(data)
        last_ts = int(data[-1]["time"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "side", "price", "qty", "usd_value"])

    df = pd.DataFrame(rows)

    # Normalise column names — Binance field names for allForceOrders
    df = df.rename(
        columns={
            "time": "timestamp_ms",
            "side": "side",
            "averagePrice": "price",
            "origQty": "qty",
        }
    )

    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["usd_value"] = df["price"] * df["qty"]
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return df[["timestamp", "side", "price", "qty", "usd_value"]]


def compute_liq_candles(liq_df: pd.DataFrame, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-event liquidations into per-candle USD volumes.

    Each candle gets:
        liq_long_usd  — USD value of SELL-side liqs (long positions force-closed)
        liq_short_usd — USD value of BUY-side liqs (short positions force-closed)
        liq_net       — liq_long_usd - liq_short_usd (positive = more longs blown)

    Args:
        liq_df: Output of fetch_liquidations().
        candles_df: OHLCV DataFrame indexed by tz-aware UTC open_time.

    Returns:
        candles_df with liq_long_usd, liq_short_usd, liq_net columns merged in.
        Candles with no liquidations have 0.0 for all three columns.
    """
    zero_cols = {"liq_long_usd": 0.0, "liq_short_usd": 0.0, "liq_net": 0.0}

    if liq_df.empty or candles_df.empty:
        out = candles_df.copy()
        for col, val in zero_cols.items():
            out[col] = val
        return out

    # Infer candle period from median gap between successive candle open times
    if len(candles_df) > 1:
        gaps = candles_df.index.to_series().diff().dropna()
        candle_period = gaps.median()
    else:
        candle_period = pd.Timedelta("5min")

    # Assign each liquidation to its candle bucket
    liq = liq_df.copy()
    liq["open_time"] = liq["timestamp"].dt.floor(candle_period)

    long_liqs = liq[liq["side"] == "SELL"].groupby("open_time")["usd_value"].sum()
    short_liqs = liq[liq["side"] == "BUY"].groupby("open_time")["usd_value"].sum()

    out = candles_df.copy()
    out["liq_long_usd"] = long_liqs.reindex(out.index).fillna(0.0).values
    out["liq_short_usd"] = short_liqs.reindex(out.index).fillna(0.0).values
    out["liq_net"] = out["liq_long_usd"] - out["liq_short_usd"]

    return out
