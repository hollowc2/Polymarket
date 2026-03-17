#!/usr/bin/env python3
"""Backtest ThreeBarMoMoStrategy on real Binance 5m BTCUSDT data.

Fetches ~90 days of candles, runs a parameter sweep on the first 75%
(train), then evaluates the best params on the held-out 25% (test).
Prints a full summary including the win rate to replace the placeholder
0.55 confidence in 3barmomo_bot.py.
"""

import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests
from polymarket_algo.backtest.engine import parameter_sweep, run_backtest, walk_forward_split
from polymarket_algo.strategies.three_bar_momo import ThreeBarMoMoStrategy

LOOKBACK_DAYS = 90
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
# Binance Vision mirror — works in regions where api.binance.com is geo-blocked
_VISION_URL = "https://data-api.binance.vision/api/v3/klines"


def _fetch_klines_vision(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """fetch_klines clone pointing at data-api.binance.vision."""
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            _VISION_URL,
            params={"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        last_open = data[-1][0]
        if last_open <= cursor:
            break
        cursor = last_open + 1
        time.sleep(0.1)

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def main() -> None:
    strategy = ThreeBarMoMoStrategy()

    # --- Fetch data ---
    now = datetime.now(tz=UTC)
    start = now - timedelta(days=LOOKBACK_DAYS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    print(f"Fetching {SYMBOL} {INTERVAL} data ({LOOKBACK_DAYS} days)...")
    candles = _fetch_klines_vision(SYMBOL, INTERVAL, start_ms, end_ms)
    candles = candles.set_index("open_time").sort_index()
    print(f"  {len(candles):,} candles loaded  ({candles.index[0].date()} → {candles.index[-1].date()})\n")

    # --- Walk-forward split ---
    train, test = walk_forward_split(candles)
    print(f"Train: {len(train):,} candles | Test: {len(test):,} candles\n")

    # --- Default params on full dataset ---
    print("=" * 60)
    print("DEFAULT PARAMS — full dataset")
    print("=" * 60)
    default_result = run_backtest(candles, strategy)
    m = default_result.metrics
    print(f"  Trade count : {m['trade_count']}")
    print(f"  Win rate    : {m['win_rate']:.1%}")
    print(f"  Total PnL   : ${m['total_pnl']:+.2f}")
    print(f"  Max drawdown: ${m['max_drawdown']:.2f}")
    print(f"  Sharpe      : {m['sharpe_ratio']:.3f}")
    print()

    # --- Parameter sweep on train set ---
    print("=" * 60)
    print("PARAMETER SWEEP — train set (top 10 by win_rate)")
    print("=" * 60)
    sweep = parameter_sweep(train, strategy, strategy.param_grid)
    top10 = sweep.head(10)
    print(
        top10[
            ["bars", "size", "size_cap", "min_body_pct", "win_rate", "total_pnl", "trade_count", "sharpe_ratio"]
        ].to_string(index=False)
    )
    print()

    # --- Best params evaluated on held-out test set ---
    best_row = sweep.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in strategy.param_grid}
    print("=" * 60)
    print(f"BEST PARAMS ON TEST SET — {best_params}")
    print("=" * 60)
    test_result = run_backtest(test, strategy, best_params)
    m = test_result.metrics
    print(f"  Trade count : {m['trade_count']}")
    print(f"  Win rate    : {m['win_rate']:.1%}")
    print(f"  Total PnL   : ${m['total_pnl']:+.2f}")
    print(f"  Max drawdown: ${m['max_drawdown']:.2f}")
    print(f"  Sharpe      : {m['sharpe_ratio']:.3f}")
    print()

    # --- Recommendation ---
    test_wr = m["win_rate"]
    print("=" * 60)
    print("CONFIDENCE RECOMMENDATION")
    print("=" * 60)
    if m["trade_count"] < 30:
        print("  WARNING: fewer than 30 test trades — sample too small, keep 0.55")
    else:
        print(f"  Measured test win rate : {test_wr:.3f}")
        print(f"  Suggested confidence   : {test_wr:.2f}  (replace 0.55 in 3barmomo_bot.py)")
        if test_wr < 0.50:
            print("  NOTE: win rate < 50% — strategy has no edge at these params on test data")


if __name__ == "__main__":
    main()
