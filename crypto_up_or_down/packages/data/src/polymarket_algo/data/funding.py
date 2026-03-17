"""Binance futures funding rate fetcher.

Funding rates are published every 8 hours. This module fetches historical
funding rate snapshots and forward-fills them to match 5-min candle frequency.

Endpoints (no auth required):
  Historical: GET https://fapi.binance.com/fapi/v1/fundingRate
  Live:       GET https://fapi.binance.com/fapi/v1/premiumIndex
"""

from __future__ import annotations

import time

import pandas as pd
import requests

FAPI_BASE = "https://fapi.binance.com"
FUNDING_RATE_URL = f"{FAPI_BASE}/fapi/v1/fundingRate"
PREMIUM_INDEX_URL = f"{FAPI_BASE}/fapi/v1/premiumIndex"
MAX_LIMIT = 1000
FUNDING_INTERVAL_MS = 8 * 3600 * 1000  # 8 hours in milliseconds


def fetch_funding_rate(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch historical funding rate snapshots from Binance futures.

    Args:
        symbol: Futures symbol, e.g. "BTCUSDT"
        start_ms: Start timestamp in milliseconds (UTC)
        end_ms: End timestamp in milliseconds (UTC)

    Returns:
        DataFrame with columns: timestamp (UTC datetime), funding_rate (float)
        One row per 8-hour funding interval. Sorted ascending by timestamp.
    """
    rows: list[dict] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        resp = requests.get(FUNDING_RATE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        rows.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= cursor:
            break
        # Advance past the last fetched funding event
        cursor = last_ts + 1
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(rows)
    df = df.rename(columns={"fundingTime": "timestamp_ms", "fundingRate": "funding_rate"})
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return df[["timestamp", "funding_rate"]]


def fetch_current_funding_rate(symbol: str) -> float | None:
    """Fetch the current (live) funding rate from Binance premiumIndex endpoint.

    Args:
        symbol: Futures symbol, e.g. "BTCUSDT"

    Returns:
        Current funding rate as a float, or None on error.
    """
    try:
        resp = requests.get(PREMIUM_INDEX_URL, params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("lastFundingRate", 0.0))
    except Exception:
        return None


def compute_funding_candles(funding_df: pd.DataFrame, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill 8-hour funding rate snapshots onto every 5-min candle.

    Also computes a rolling 30-day z-score of the funding rate for use
    as a signal normalization (funding_zscore).

    Args:
        funding_df: Output of fetch_funding_rate(). Must have columns:
                    timestamp (tz-aware UTC), funding_rate (float).
        candles_df: OHLCV DataFrame indexed by tz-aware UTC open_time.

    Returns:
        candles_df with two extra columns merged in:
          - funding_rate: forward-filled funding rate at each candle
          - funding_zscore: rolling 30-day z-score of funding_rate
        Rows without a funding predecessor are filled with 0.
    """
    if funding_df.empty or candles_df.empty:
        out = candles_df.copy()
        out["funding_rate"] = 0.0
        out["funding_zscore"] = 0.0
        return out

    # Build a Series indexed by funding timestamps
    funding_series = funding_df.set_index("timestamp")["funding_rate"]
    funding_series = funding_series[~funding_series.index.duplicated(keep="first")]
    funding_series = funding_series.sort_index()

    # Reindex to candle frequency using forward-fill
    combined_index = funding_series.index.union(candles_df.index)
    funding_reindexed = funding_series.reindex(combined_index).ffill().bfill()
    funding_at_candles = funding_reindexed.reindex(candles_df.index).fillna(0.0)

    out = candles_df.copy()
    out["funding_rate"] = funding_at_candles.values

    # Compute z-score at 8-hour frequency (one value per funding snapshot) then forward-fill.
    # Computing z-score on forward-filled 5m data produces a noisy step-function because
    # the same value repeats ~96 times per 8h window, making rolling std unstable at
    # boundaries and generating spurious extremes.
    #
    # Correct approach:
    #   1. z-score on the raw 8h funding series (30-day rolling = ~90 snapshots)
    #   2. Forward-fill z-score to every candle (changes only at 8h boundaries)
    #
    # Window: 30 days × 3 snapshots/day = 90 snapshots
    ZSCORE_WINDOW = 90  # 8h snapshots

    roll_mean_8h = funding_series.rolling(ZSCORE_WINDOW, min_periods=10).mean()
    roll_std_8h = funding_series.rolling(ZSCORE_WINDOW, min_periods=10).std()
    zscore_8h = (funding_series - roll_mean_8h) / roll_std_8h.replace(0.0, float("nan"))
    zscore_8h = zscore_8h.fillna(0.0)

    # Forward-fill z-score from 8h timestamps to every candle index
    combined_index = zscore_8h.index.union(candles_df.index)
    zscore_reindexed = zscore_8h.reindex(combined_index).ffill().bfill()
    zscore_at_candles = zscore_reindexed.reindex(candles_df.index).fillna(0.0)
    out["funding_zscore"] = zscore_at_candles.values

    return out
