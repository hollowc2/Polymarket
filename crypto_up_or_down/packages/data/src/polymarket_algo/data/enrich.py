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

    # With optional new enrichments:
    enriched = enrich_candles(
        candles, symbol="BTCUSDT",
        include_basis=True,     # basis, basis_zscore
        include_oi=True,        # oi, oi_roc, oi_zscore
        include_coinbase=True,  # coinbase_premium, coinbase_premium_zscore
        include_deribit=True,   # iv_atm, iv_zscore, skew_25d
    )
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
    include_basis: bool = False,
    include_oi: bool = False,
    include_coinbase: bool = False,
    include_deribit: bool = False,
) -> pd.DataFrame:
    """Fetch microstructure data and merge it onto a candle DataFrame.

    Args:
        candles: OHLCV DataFrame indexed by tz-aware UTC open_time.
                 Must contain at least a 'close' column.
        symbol:  Binance symbol string, e.g. "BTCUSDT".
                 For futures endpoints (liq/funding/basis/oi) this is used as-is.
        include_cvd:     Fetch aggTrades and add delta/cvd columns.
        include_liq:     Fetch force orders and add liq_long/short_usd columns.
        include_funding: Fetch funding rate and add funding_rate/zscore columns.
        include_basis:   Fetch perp OHLCV and add basis/basis_zscore columns.
        include_oi:      Fetch OI history and add oi/oi_roc/oi_zscore columns.
        include_coinbase: Fetch Coinbase candles and add coinbase_premium/zscore.
        include_deribit: Fetch Deribit DVOL and add iv_atm/iv_zscore/skew_25d.

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

    # --- Spot-perp basis ---
    if include_basis:
        try:
            from polymarket_algo.data.binance import compute_basis_candles, fetch_perp_klines

            # Infer interval from candle spacing
            if len(result) >= 2:
                spacing_s = (result.index[1] - result.index[0]).total_seconds()
                interval = _seconds_to_interval(int(spacing_s))
            else:
                interval = "5m"

            perp_df = fetch_perp_klines(symbol, interval, start_ms, end_ms)
            # Set index to open_time for compute_basis_candles
            if not perp_df.empty:
                perp_df = perp_df.set_index("open_time").sort_index()
            result = compute_basis_candles(perp_df, result)
        except Exception as exc:
            print(f"[enrich] Basis fetch failed ({exc}), using zeros")
            result["basis"] = 0.0
            result["basis_zscore"] = 0.0

    # --- Open interest ---
    if include_oi:
        try:
            from polymarket_algo.data.binance import compute_oi_candles, fetch_open_interest_hist

            if len(result) >= 2:
                spacing_s = (result.index[1] - result.index[0]).total_seconds()
                oi_period = _seconds_to_oi_period(int(spacing_s))
            else:
                oi_period = "5m"

            oi_df = fetch_open_interest_hist(symbol, oi_period, start_ms, end_ms)
            result = compute_oi_candles(oi_df, result)
        except Exception as exc:
            print(f"[enrich] OI fetch failed ({exc}), using zeros")
            result["oi"] = 0.0
            result["oi_roc"] = 0.0
            result["oi_zscore"] = 0.0

    # --- Coinbase premium ---
    if include_coinbase:
        try:
            from polymarket_algo.data.coinbase import compute_premium_candles, fetch_coinbase_klines

            if len(result) >= 2:
                spacing_s = (result.index[1] - result.index[0]).total_seconds()
                interval = _seconds_to_interval(int(spacing_s))
            else:
                interval = "5m"

            cb_df = fetch_coinbase_klines(symbol, interval, start_ms, end_ms)
            result = compute_premium_candles(cb_df, result)
        except Exception as exc:
            print(f"[enrich] Coinbase premium fetch failed ({exc}), using zeros")
            result["coinbase_premium"] = 0.0
            result["coinbase_premium_zscore"] = 0.0

    # --- Deribit DVOL ---
    if include_deribit:
        try:
            from polymarket_algo.data.deribit import compute_deribit_candles, fetch_dvol_hourly

            iv_df = fetch_dvol_hourly(symbol, start_ms, end_ms)
            result = compute_deribit_candles(iv_df, result)
        except Exception as exc:
            print(f"[enrich] Deribit DVOL fetch failed ({exc}), using zeros")
            result["iv_atm"] = 0.0
            result["iv_zscore"] = 0.0
            result["skew_25d"] = 0.0

    return result


def _seconds_to_interval(seconds: int) -> str:
    """Map candle spacing in seconds to a Binance/Coinbase interval string."""
    mapping = {60: "1m", 300: "5m", 900: "15m", 1800: "30m", 3600: "1h", 14400: "4h", 86400: "1d"}
    return mapping.get(seconds, "5m")


def _seconds_to_oi_period(seconds: int) -> str:
    """Map candle spacing in seconds to a Binance OI period string."""
    mapping = {300: "5m", 900: "15m", 1800: "30min", 3600: "1h", 7200: "2h", 14400: "4h", 86400: "1d"}
    return mapping.get(seconds, "5m")
