"""Candle enrichment pipeline.

Merges non-TA microstructure data into an OHLCV candle DataFrame before
strategy evaluation. Strategies just read enriched columns — they don't
fetch their own data.

Usage:
    from polymarket_algo.data import fetch_klines, enrich_candles

    candles = fetch_klines("BTCUSDT", "5m", start_ms, end_ms)
    candles = candles.set_index("open_time").sort_index()
    enriched = enrich_candles(candles, symbol="BTCUSDT")

    # enriched now has: buy_vol, sell_vol, delta, cvd,
    #                   liq_long_usd, liq_short_usd, liq_net,
    #                   funding_rate, funding_zscore
"""

from __future__ import annotations

import pandas as pd
from polymarket_algo.data.agg_trades import compute_cvd_candles, fetch_agg_trades
from polymarket_algo.data.funding import compute_funding_candles, fetch_funding_rate
from polymarket_algo.data.liquidations import compute_liq_candles, fetch_liquidations


def enrich_candles(
    candles: pd.DataFrame,
    symbol: str = "BTCUSDT",
    *,
    include_cvd: bool = True,
    include_liq: bool = True,
    include_funding: bool = True,
) -> pd.DataFrame:
    """Fetch microstructure data and merge it onto a candle DataFrame.

    Args:
        candles: OHLCV DataFrame indexed by tz-aware UTC open_time.
                 Must contain at least a 'close' column.
        symbol:  Binance symbol string, e.g. "BTCUSDT".
                 For futures endpoints (liq/funding) this is used as-is.
        include_cvd:     Fetch aggTrades and add delta/cvd columns.
        include_liq:     Fetch force orders and add liq_long/short_usd columns.
        include_funding: Fetch funding rate and add funding_rate/zscore columns.

    Returns:
        New DataFrame (copy of candles) with extra columns merged in.
        Missing enrichments get neutral/zero defaults so strategies
        degrade gracefully rather than raising KeyError.

    Note:
        CVD enrichment is the heaviest (thousands of API calls for long
        date ranges). For quick experimentation, set include_cvd=False.
    """
    if candles.empty:
        return candles.copy()

    # Derive time range from candle index
    start_ms = int(candles.index.min().timestamp() * 1000)
    end_ms = int(candles.index.max().timestamp() * 1000) + 1

    result = candles.copy()

    # --- CVD (aggTrades) ---
    if include_cvd:
        try:
            trades_df = fetch_agg_trades(symbol, start_ms, end_ms)
            result = compute_cvd_candles(trades_df, result)
        except Exception as exc:
            print(f"[enrich] CVD fetch failed ({exc}), using zeros")
            result["buy_vol"] = 0.0
            result["sell_vol"] = 0.0
            result["delta"] = 0.0
            result["cvd"] = 0.0

    # --- Liquidations ---
    if include_liq:
        try:
            liq_df = fetch_liquidations(symbol, start_ms, end_ms)
            result = compute_liq_candles(liq_df, result)
        except Exception as exc:
            print(f"[enrich] Liquidations fetch failed ({exc}), using zeros")
            result["liq_long_usd"] = 0.0
            result["liq_short_usd"] = 0.0
            result["liq_net"] = 0.0

    # --- Funding rate ---
    if include_funding:
        try:
            funding_df = fetch_funding_rate(symbol, start_ms, end_ms)
            result = compute_funding_candles(funding_df, result)
        except Exception as exc:
            print(f"[enrich] Funding rate fetch failed ({exc}), using zeros")
            result["funding_rate"] = 0.0
            result["funding_zscore"] = 0.0

    return result
