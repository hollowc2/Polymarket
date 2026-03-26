"""Coinbase Exchange spot OHLCV fetcher + premium index computation.

Fetches BTC-USD / ETH-USD candles from the Coinbase Exchange public API
(no authentication required) and computes the Coinbase premium index
relative to Binance prices.

Coinbase Premium Index:
    premium = (coinbase_close - binance_close) / binance_close

A positive premium means US buyers are paying more than the global Binance
price — a bullish signal. A negative premium means US sellers are more
aggressive than the global market — a bearish signal.

Endpoints (no auth):
    https://api.exchange.coinbase.com/products/{product_id}/candles
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pandas as pd
import requests

COINBASE_BASE_URL = "https://api.exchange.coinbase.com"
COINBASE_CANDLES_URL = f"{COINBASE_BASE_URL}/products/{{product_id}}/candles"
COINBASE_MAX_CANDLES = 300  # Hard limit per request

# Granularity in seconds: Coinbase only accepts specific values
COINBASE_GRANULARITY: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}

# Binance symbol → Coinbase product_id
SYMBOL_MAP: dict[str, str] = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
}

ZSCORE_WINDOW = 20  # bars for rolling z-score


def _binance_to_coinbase(symbol: str) -> str:
    """Convert Binance symbol (BTCUSDT) to Coinbase product_id (BTC-USD)."""
    product = SYMBOL_MAP.get(symbol)
    if product is None:
        raise ValueError(f"No Coinbase mapping for symbol {symbol!r}. Supported: {list(SYMBOL_MAP)}")
    return product


def fetch_coinbase_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch OHLCV candles from Coinbase Exchange public API.

    Args:
        symbol:    Binance-style symbol, e.g. "BTCUSDT". Converted to Coinbase format.
        interval:  Candle interval, e.g. "5m", "15m", "1h".
        start_ms:  Start time in milliseconds (UTC).
        end_ms:    End time in milliseconds (UTC).

    Returns:
        DataFrame with columns: open_time (tz-aware UTC datetime index), open,
        high, low, close, volume. Sorted ascending by open_time.
    """
    product_id = _binance_to_coinbase(symbol)
    granularity_sec = COINBASE_GRANULARITY.get(interval)
    if granularity_sec is None:
        raise ValueError(f"Unsupported interval for Coinbase: {interval!r}. Supported: {list(COINBASE_GRANULARITY)}")

    url = COINBASE_CANDLES_URL.format(product_id=product_id)
    rows: list[list] = []

    # Walk forward: Coinbase returns newest-first; paginate by advancing start
    cursor_sec = start_ms // 1000
    end_sec = end_ms // 1000

    while cursor_sec < end_sec:
        page_end_sec = min(cursor_sec + COINBASE_MAX_CANDLES * granularity_sec, end_sec)
        params = {
            "start": datetime.fromtimestamp(cursor_sec, tz=UTC).isoformat(),
            "end": datetime.fromtimestamp(page_end_sec, tz=UTC).isoformat(),
            "granularity": granularity_sec,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json()

        if not page:
            break

        rows.extend(page)  # newest first within page, but pages walk forward
        last_ts = int(page[-1][0])  # oldest in this page (last element = smallest ts)
        if last_ts <= cursor_sec:
            break
        cursor_sec = page_end_sec + granularity_sec
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    # Coinbase row format: [time_sec, low, high, open, close, volume]
    df = pd.DataFrame(rows, columns=["time_sec", "low", "high", "open", "close", "volume"])
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df.dropna(subset=["time_sec"])
    df = df.drop_duplicates(subset=["time_sec"]).sort_values("time_sec").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["time_sec"] * 1000, unit="ms", utc=True)

    return df[["open_time", "open", "high", "low", "close", "volume"]]


def compute_premium_candles(coinbase_df: pd.DataFrame, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Coinbase premium index and merge onto Binance candles.

    The premium index measures how much more (or less) buyers pay on
    Coinbase vs Binance. A positive premium signals US institutional/retail
    buying pressure; a negative premium signals US selling pressure.

    Args:
        coinbase_df: Output of fetch_coinbase_klines(). Must have columns:
                     open_time (tz-aware UTC) and close (float).
        candles_df:  Binance OHLCV DataFrame indexed by tz-aware UTC open_time.
                     Must contain a 'close' column.

    Returns:
        candles_df copy with two extra columns:
          - coinbase_premium: (cb_close - binance_close) / binance_close
          - coinbase_premium_zscore: rolling z-score (window=ZSCORE_WINDOW bars)
        Rows without a Coinbase price match default to 0.0.
    """
    out = candles_df.copy()

    if coinbase_df.empty or candles_df.empty:
        out["coinbase_premium"] = 0.0
        out["coinbase_premium_zscore"] = 0.0
        return out

    # Align Coinbase close onto candle index via forward-fill
    cb_series = coinbase_df.set_index("open_time")["close"]
    cb_series = cb_series[~cb_series.index.duplicated(keep="first")].sort_index()

    combined_idx = cb_series.index.union(candles_df.index)
    cb_reindexed = cb_series.reindex(combined_idx).ffill().bfill()
    cb_at_candles = cb_reindexed.reindex(candles_df.index).fillna(float("nan"))

    binance_close = candles_df["close"]
    premium = (cb_at_candles - binance_close) / binance_close.replace(0, float("nan"))
    premium = premium.fillna(0.0)

    roll_mean = premium.rolling(ZSCORE_WINDOW, min_periods=5).mean()
    roll_std = premium.rolling(ZSCORE_WINDOW, min_periods=5).std()
    zscore = (premium - roll_mean) / roll_std.replace(0.0, float("nan"))
    zscore = zscore.fillna(0.0)

    out["coinbase_premium"] = premium.values
    out["coinbase_premium_zscore"] = zscore.values
    return out
