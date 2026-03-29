"""Deribit implied volatility data fetcher.

Fetches historical implied volatility (DVOL index) from Deribit's public REST API
and computes a rolling z-score for use as a regime indicator in strategies.

The Deribit Volatility Index (DVOL) is analogous to the VIX for crypto — it
represents the 30-day implied vol derived from BTC/ETH options prices. High DVOL
= elevated uncertainty; low DVOL = calm market.

For directional strategies, DVOL is used as a GATE (skip trades when vol is too
high or moving too fast) rather than a directional signal itself.

Endpoints (no auth required):
    Historical DVOL: GET https://www.deribit.com/api/v2/public/get_historical_volatility
    High-freq DVOL:  GET https://www.deribit.com/api/v2/public/get_volatility_index_data

Note on skew:
    25-delta put/call risk reversal requires fetching individual options order books
    in real time — there is no historical free endpoint. The DeribitSkewStrategy
    fetches live skew internally. For backtesting, skew defaults to 0.0 and the
    strategy falls back to IV-only signals.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
DVOL_HIST_URL = f"{DERIBIT_BASE}/get_historical_volatility"
DVOL_INDEX_URL = f"{DERIBIT_BASE}/get_volatility_index_data"
INSTRUMENTS_URL = f"{DERIBIT_BASE}/get_instruments"
BOOK_SUMMARY_URL = f"{DERIBIT_BASE}/get_book_summary_by_instrument"

ZSCORE_WINDOW = 20  # bars for rolling iv z-score

# Deribit currency names (not Binance symbols)
DERIBIT_CURRENCY: dict[str, str] = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "BTC": "BTC",
    "ETH": "ETH",
}


def _to_deribit_currency(symbol: str) -> str:
    currency = DERIBIT_CURRENCY.get(symbol.upper())
    if currency is None:
        raise ValueError(f"No Deribit currency mapping for {symbol!r}. Supported: {list(DERIBIT_CURRENCY)}")
    return currency


def fetch_historical_volatility(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch Deribit DVOL historical data at daily resolution.

    Uses the get_historical_volatility endpoint which returns all historical data
    (no pagination needed). Filters to the requested time range.

    Args:
        symbol:   Binance symbol or Deribit currency: "BTCUSDT" / "BTC" / "ETH"
        start_ms: Start time in milliseconds (UTC)
        end_ms:   End time in milliseconds (UTC)

    Returns:
        DataFrame with columns:
          - timestamp (tz-aware UTC datetime)
          - iv_atm (float): annualized implied vol as a percentage (e.g. 60.0 = 60%)
        One row per day. Sorted ascending.
    """
    currency = _to_deribit_currency(symbol)
    try:
        resp = requests.get(DVOL_HIST_URL, params={"currency": currency}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[deribit] Historical volatility fetch failed: {exc}")
        return pd.DataFrame(columns=["timestamp", "iv_atm"])

    result = data.get("result", [])
    if not result:
        return pd.DataFrame(columns=["timestamp", "iv_atm"])

    df = pd.DataFrame(result, columns=["timestamp_ms", "iv_atm"])
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["iv_atm"] = pd.to_numeric(df["iv_atm"], errors="coerce")
    df = df.dropna()
    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)]
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df[["timestamp", "iv_atm"]]


def fetch_dvol_hourly(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch Deribit DVOL at hourly resolution for more granular IV data.

    Uses the get_volatility_index_data endpoint with resolution=3600 (1 hour).
    Paginates if needed (max 744 hours = 31 days per request).

    Args:
        symbol:   Binance symbol or Deribit currency
        start_ms: Start time in milliseconds (UTC)
        end_ms:   End time in milliseconds (UTC)

    Returns:
        DataFrame with columns: timestamp (tz-aware UTC), iv_atm (float).
        One row per hour. Sorted ascending.
    """
    currency = _to_deribit_currency(symbol)
    resolution = 3600  # 1 hour in seconds
    max_points = 744   # ~31 days of hourly data per request
    rows: list[dict] = []
    cursor_ms = start_ms

    while cursor_ms < end_ms:
        page_end_ms = min(cursor_ms + max_points * resolution * 1000, end_ms)
        try:
            resp = requests.get(
                DVOL_INDEX_URL,
                params={
                    "currency": currency,
                    "start_timestamp": cursor_ms,
                    "end_timestamp": page_end_ms,
                    "resolution": resolution,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[deribit] DVOL hourly fetch failed: {exc}")
            break

        result = data.get("result", {})
        timestamps = result.get("timestamps", [])
        open_vals = result.get("open", [])

        if not timestamps:
            break

        for ts, val in zip(timestamps, open_vals, strict=False):
            rows.append({"timestamp_ms": ts, "iv_atm": val})

        last_ts = int(timestamps[-1])
        if last_ts <= cursor_ms:
            break
        cursor_ms = last_ts + resolution * 1000
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "iv_atm"])

    df = pd.DataFrame(rows)
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["iv_atm"] = pd.to_numeric(df["iv_atm"], errors="coerce")
    df = df.dropna()
    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)]
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df[["timestamp", "iv_atm"]]


def fetch_live_skew(symbol: str) -> float | None:
    """Fetch approximate 25-delta put/call risk reversal from Deribit (live only).

    Finds the nearest-expiry options (7-30 days out) and approximates skew as
    the difference between put and call implied volatility at ATM strikes.

    Returns:
        Approximate skew as a float (positive = calls more expensive = bullish),
        or None if the fetch fails.

    Note:
        This is a live-only function. There is no historical skew endpoint in
        the free Deribit API. For accurate 25-delta skew, use the Deribit private
        API with proper options delta calculation. This approximation uses ATM IVs.
    """
    currency = _to_deribit_currency(symbol)
    try:
        # Get active options instruments
        resp = requests.get(
            INSTRUMENTS_URL,
            params={"currency": currency, "kind": "option", "expired": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        instruments = resp.json().get("result", [])
    except Exception as exc:
        print(f"[deribit] Instruments fetch failed: {exc}")
        return None

    if not instruments:
        return None

    # Filter to options expiring 7-30 days from now, European style
    import time as time_mod
    now_ms = int(time_mod.time() * 1000)
    min_expiry = now_ms + 7 * 86400 * 1000
    max_expiry = now_ms + 30 * 86400 * 1000

    candidates = [
        inst for inst in instruments
        if min_expiry <= inst.get("expiration_timestamp", 0) <= max_expiry
        and inst.get("option_type") in ("call", "put")
    ]
    if not candidates:
        return None

    # Group by expiry, find nearest expiry
    expiries = sorted({inst["expiration_timestamp"] for inst in candidates})
    nearest_expiry = expiries[0]
    expiry_opts = [inst for inst in candidates if inst["expiration_timestamp"] == nearest_expiry]

    # Separate calls and puts
    calls = {inst["strike"]: inst["instrument_name"] for inst in expiry_opts if inst["option_type"] == "call"}
    puts = {inst["strike"]: inst["instrument_name"] for inst in expiry_opts if inst["option_type"] == "put"}

    # Find strikes common to both, then pick nearest to spot price
    common_strikes = sorted(set(calls) & set(puts))
    if not common_strikes:
        return None

    # Fetch Deribit index price to find nearest-to-spot strike (true ATM proxy)
    spot_price: float | None = None
    try:
        idx_resp = requests.get(
            f"{DERIBIT_BASE}/get_index_price",
            params={"index_name": f"{currency.lower()}_usd"},
            timeout=10,
        )
        idx_resp.raise_for_status()
        spot_price = idx_resp.json().get("result", {}).get("index_price")
    except Exception:
        pass

    if spot_price is not None:
        atm_strike = min(common_strikes, key=lambda s: abs(s - spot_price))
    else:
        atm_strike = common_strikes[len(common_strikes) // 2]
    call_name = calls.get(atm_strike)
    put_name = puts.get(atm_strike)
    if not call_name or not put_name:
        return None

    try:
        call_resp = requests.get(BOOK_SUMMARY_URL, params={"instrument_name": call_name}, timeout=10)
        put_resp = requests.get(BOOK_SUMMARY_URL, params={"instrument_name": put_name}, timeout=10)
        call_resp.raise_for_status()
        put_resp.raise_for_status()
        call_result = call_resp.json().get("result", [{}])
        put_result = put_resp.json().get("result", [{}])
        call_iv = call_result[0].get("mark_iv") if call_result else None
        put_iv = put_result[0].get("mark_iv") if put_result else None
    except Exception as exc:
        print(f"[deribit] Skew fetch failed: {exc}")
        return None

    if call_iv is None or put_iv is None:
        return None

    # Skew: positive = calls more expensive (bullish consensus)
    # Negative = puts more expensive (bearish consensus)
    return float(call_iv) - float(put_iv)


def compute_deribit_candles(iv_df: pd.DataFrame, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill Deribit DVOL data onto candles and compute rolling z-score.

    Args:
        iv_df:      Output of fetch_historical_volatility() or fetch_dvol_hourly().
                    Must have columns: timestamp (tz-aware UTC), iv_atm (float).
        candles_df: OHLCV DataFrame indexed by tz-aware UTC open_time.

    Returns:
        candles_df copy with extra columns:
          - iv_atm: forward-filled DVOL value at each candle
          - iv_zscore: rolling z-score of iv_atm (window=ZSCORE_WINDOW)
          - skew_25d: 0.0 (historical not available; fetched live by strategy)
        Missing DVOL data defaults to 0.0.
    """
    out = candles_df.copy()

    if iv_df.empty or candles_df.empty:
        out["iv_atm"] = 0.0
        out["iv_zscore"] = 0.0
        out["skew_25d"] = 0.0
        return out

    iv_series = iv_df.set_index("timestamp")["iv_atm"]
    iv_series = iv_series[~iv_series.index.duplicated(keep="first")].sort_index()

    combined_idx = iv_series.index.union(candles_df.index)
    iv_reindexed = iv_series.reindex(combined_idx).ffill().bfill()
    iv_at_candles = iv_reindexed.reindex(candles_df.index).fillna(0.0)

    roll_mean = iv_at_candles.rolling(ZSCORE_WINDOW, min_periods=5).mean()
    roll_std = iv_at_candles.rolling(ZSCORE_WINDOW, min_periods=5).std()
    zscore = (iv_at_candles - roll_mean) / roll_std.replace(0.0, float("nan"))
    zscore = zscore.fillna(0.0)

    out["iv_atm"] = iv_at_candles.values
    out["iv_zscore"] = zscore.values
    out["skew_25d"] = 0.0  # Historical skew not available; set live in strategy bot
    return out
