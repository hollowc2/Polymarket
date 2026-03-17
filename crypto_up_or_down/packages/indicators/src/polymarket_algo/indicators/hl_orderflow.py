"""
HLOrderFlow — Hyperliquid perpetuals order flow signal.

Fetches OHLCV candles from the Hyperliquid public REST API and computes
buy/sell pressure across timeframes. No API key required.

Method: candles where close >= open are classified as buy volume;
close < open as sell volume. This is a momentum proxy, not true CVD.
Most reliable on short windows (5m, 15m). Noisier on 4h+.

Supported coins: BTC, ETH, HYPE, SOL, XRP
Supported timeframes: 5m, 15m, 1h, 4h (for window signals)

--- VPS / multi-process deployment ---
The default in-memory cache is per-process. If you run multiple workers
(e.g. via docker-compose or process-compose), each worker fetches
independently and you'll hit HL rate limits faster.

To share the cache across processes, set the environment variable:

    HL_ORDERFLOW_CACHE_FILE=/tmp/hl_orderflow_cache.json

All processes pointing at the same file will share a single fetch per
TTL window. The file is written atomically (temp file + rename).
If the file is stale or missing, the process fetches fresh and writes it.
"""

import json
import os
import tempfile
import time

import requests

_HL_URL = "https://api.hyperliquid.xyz/info"

_TIMEFRAME_MAP = {
    "5m": ("5m", 300_000),
    "15m": ("15m", 900_000),
    "1h": ("1h", 3_600_000),
    "4h": ("4h", 14_400_000),
}

# L1: in-process memory cache
_mem_cache: dict = {}

# L2: optional shared file cache (set HL_ORDERFLOW_CACHE_FILE env var)
_CACHE_FILE = os.environ.get("HL_ORDERFLOW_CACHE_FILE", "")

_TTL = 60  # seconds


# ---------------------------------------------------------------------------
# Transport — retry with exponential backoff
# ---------------------------------------------------------------------------


def _hl_post(payload: dict, retries: int = 3) -> list | dict:
    """POST to HL public API with retry on 429/5xx/timeout."""
    delay = 1.0
    last_exc = None
    for _attempt in range(retries):
        try:
            resp = requests.post(_HL_URL, json=payload, timeout=10)
            if resp.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            time.sleep(delay)
            delay *= 2
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise  # 4xx other than 429 — don't retry
            last_exc = exc
            time.sleep(delay)
            delay *= 2
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"HL API unreachable after {retries} attempts") from last_exc


# ---------------------------------------------------------------------------
# Cache — L1 memory, L2 shared file
# ---------------------------------------------------------------------------


def _file_cache_read(key: str) -> tuple[float, dict] | None:
    """Read a single key from the shared file cache. Returns (inserted_at, data) or None."""
    if not _CACHE_FILE:
        return None
    try:
        with open(_CACHE_FILE) as f:
            store = json.load(f)
        entry = store.get(key)
        if entry:
            return entry["inserted_at"], entry["data"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


def _file_cache_write(key: str, inserted_at: float, data: dict) -> None:
    """Write a single key to the shared file cache atomically."""
    if not _CACHE_FILE:
        return
    try:
        try:
            with open(_CACHE_FILE) as f:
                store = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            store = {}
        store[key] = {"inserted_at": inserted_at, "data": data}
        dir_ = os.path.dirname(_CACHE_FILE) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(store, f)
            os.replace(tmp_path, _CACHE_FILE)
        except Exception:
            os.unlink(tmp_path)
            raise
    except Exception:
        pass  # file cache failure is non-fatal; fall back to mem cache


def _cached(key: str, fn):
    now = time.time()

    # L1 — memory
    if key in _mem_cache:
        inserted_at, data = _mem_cache[key]
        if now - inserted_at < _TTL:
            return data

    # L2 — shared file
    file_entry = _file_cache_read(key)
    if file_entry:
        inserted_at, data = file_entry
        if now - inserted_at < _TTL:
            _mem_cache[key] = (inserted_at, data)  # warm L1
            return data

    # fetch fresh
    result = fn()
    _mem_cache[key] = (now, result)
    _file_cache_write(key, now, result)
    return result


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _orderflow_for_coin(coin: str, tf: str) -> dict:
    """Buy/sell pressure for one coin over one timeframe window."""
    resolution, lookback_ms = _TIMEFRAME_MAP[tf]
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - lookback_ms

    candles = _hl_post(
        {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": resolution, "startTime": start_ms, "endTime": now_ms},
        }
    )

    buy_vol = sell_vol = 0.0
    for c in candles or []:
        vol = float(c.get("v", 0))
        if float(c.get("c", 0)) >= float(c.get("o", 0)):
            buy_vol += vol
        else:
            sell_vol += vol

    total = buy_vol + sell_vol
    pressure = (buy_vol / total) if total > 0 else 0.5
    return {
        "buy_pressure": pressure,
        "sell_pressure": 1 - pressure,
        "cumulative_delta": buy_vol - sell_vol,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "dominant_side": "BUY" if pressure > 0.55 else ("SELL" if pressure < 0.45 else "NEUTRAL"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hl_orderflow(coin: str = "BTC") -> dict:
    """
    Return buy/sell pressure signals for a coin across 5m, 15m, 1h, 4h windows.

    Results are cached for 60s. On VPS with multiple workers, set
    HL_ORDERFLOW_CACHE_FILE=/tmp/hl_orderflow_cache.json to share the cache.

    Returns:
        {
            "5m":  { buy_pressure, sell_pressure, cumulative_delta, dominant_side, ... },
            "15m": { ... },
            "1h":  { ... },
            "4h":  { ... },
        }

    Raises RuntimeError if HL API is unreachable after 3 retries.
    """
    cache_key = f"hl_orderflow_{coin}"

    def _fetch():
        return {tf: _orderflow_for_coin(coin, tf) for tf in _TIMEFRAME_MAP}

    return _cached(cache_key, _fetch)


def hl_orderflow_signal(coin: str = "BTC", tf: str = "5m") -> str:
    """
    Convenience function. Returns "BUY", "SELL", or "NEUTRAL" for a coin/timeframe.

    Args:
        coin: BTC, ETH, HYPE, SOL, or XRP
        tf:   5m, 15m, 1h, or 4h

    Returns "NEUTRAL" on API failure (does not propagate exceptions).

    Example:
        signal = hl_orderflow_signal("BTC", "5m")
        if signal == "BUY":
            # favour long / YES side
    """
    try:
        data = hl_orderflow(coin)
        return data.get(tf, {}).get("dominant_side", "NEUTRAL")
    except Exception:
        return "NEUTRAL"


class HLOrderFlowIndicator:
    """
    Hyperliquid order flow indicator for use in polymarket-algo strategies.

    Entry point name: hl_orderflow

    compute() returns the full signal dict across all timeframes for the
    requested coin. Use the 5m and 15m windows for short-term signals.

    Params:
        coin (str): BTC, ETH, HYPE, SOL, XRP  (default: BTC)

    Returns:
        dict keyed by timeframe ("5m", "15m", "1h", "4h"), each containing:
            buy_pressure   (float 0–1)
            sell_pressure  (float 0–1)
            cumulative_delta (float, buy_vol - sell_vol in coin units)
            dominant_side  ("BUY" | "SELL" | "NEUTRAL")
            buy_volume     (float)
            sell_volume    (float)

    Returns all NEUTRAL values on API failure rather than raising.

    Example:
        indicator = HLOrderFlowIndicator()
        signals = indicator.compute(coin="BTC")

        if signals["5m"]["dominant_side"] == "BUY" and signals["15m"]["buy_pressure"] > 0.6:
            # short-term buy signal — consider YES on a BTC price market

    VPS deployment:
        Set HL_ORDERFLOW_CACHE_FILE=/tmp/hl_orderflow_cache.json in your
        environment to share the 60s cache across all worker processes.
    """

    name = "hl_orderflow"

    _NEUTRAL = {
        tf: {
            "buy_pressure": 0.5,
            "sell_pressure": 0.5,
            "cumulative_delta": 0.0,
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "dominant_side": "NEUTRAL",
        }
        for tf in _TIMEFRAME_MAP
    }

    def compute(self, series=None, **params):
        coin = params.get("coin", "BTC")
        try:
            return hl_orderflow(coin=coin)
        except Exception:
            return self._NEUTRAL
