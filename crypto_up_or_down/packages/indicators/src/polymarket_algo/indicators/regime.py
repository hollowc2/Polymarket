"""BTC regime gate: ATR percentile rank on 1h candles.

Returns True (trending — trade allowed) when the current ATR sits at or above
atr_pct_floor within the recent lookback window.  Falls open (True) on any
data error so a fetch failure never silently blocks the bot.

Usage:
    from polymarket_algo.indicators.regime import regime_ok
    if not regime_ok(atr_pct_floor=25.0):
        log("Ranging regime — skip")
"""

from __future__ import annotations

import time

import pandas as pd


def regime_ok(
    coin: str = "BTC",
    atr_period: int = 14,
    lookback_hours: int = 72,
    atr_pct_floor: float = 25.0,
) -> bool:
    """Return True if BTC is in a trending regime.

    Trending = current ATR(14) on 1h candles ranks at or above atr_pct_floor
    percentile within the recent lookback window.

    Args:
        coin: Binance base coin (default BTC → BTCUSDT).
        atr_period: ATR smoothing period (default 14).
        lookback_hours: Hours of 1h candles to fetch for the percentile window.
        atr_pct_floor: Minimum ATR percentile rank (0–100). Below this = ranging.

    Returns True on any data failure (fail-open — don't block on bad data).
    """
    try:
        from polymarket_algo.data.binance import fetch_klines

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - lookback_hours * 60 * 60 * 1000
        candles = fetch_klines(f"{coin}USDT", "1h", start_ms, now_ms)

        if candles.empty or len(candles) < atr_period + 5:
            return True

        high = candles["high"]
        low = candles["low"]
        prev_close = candles["close"].shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(atr_period, min_periods=atr_period).mean().dropna()

        if atr.empty:
            return True

        current_atr = float(atr.iloc[-1])
        pct_rank = float((atr < current_atr).mean() * 100)
        return pct_rank >= atr_pct_floor

    except Exception:
        return True
