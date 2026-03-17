#!/usr/bin/env python3
"""Backtest StreakReversalStrategy on real Binance 5m BTCUSDT data.

Fetches ~90 days of candles, runs a full param sweep on the train set,
then evaluates best params on the held-out test set.

Also cross-references results against the hardcoded REVERSAL_RATES table
in sizing.py (which was derived from Polymarket outcomes, not Binance data)
to see whether those historical rates hold up on raw price data.
"""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from polymarket_algo.backtest.engine import parameter_sweep, run_backtest, walk_forward_split
from polymarket_algo.core.sizing import REVERSAL_RATES, get_rate_estimate
from polymarket_algo.strategies.streak_reversal import StreakReversalStrategy

LOOKBACK_DAYS = 730
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
_VISION_URL = "https://data-api.binance.vision/api/v3/klines"


def _fetch_klines_vision(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
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


def win_rate_by_trigger(candles: pd.DataFrame, strategy: StreakReversalStrategy) -> None:
    """Print win rate + 95% Wilson CI for each trigger length."""
    import math

    def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """95% Wilson score confidence interval for a proportion."""
        if n == 0:
            return 0.0, 0.0
        p = wins / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return centre - margin, centre + margin

    asset = SYMBOL.replace("USDT", "")

    print("=" * 60)
    print("WIN RATE BY TRIGGER LENGTH — full dataset")
    print(f"  95% Wilson CI  |  cross-ref vs Polymarket rates ({asset})")
    print("=" * 60)
    hdr = f"  {'trig':>4}  {'trades':>7}  {'win_rate':>9}  {'95% CI':>15}"
    hdr += f"  {'±':>6}  {'poly':>6}  {'poly CI':>15}  {'delta':>7}"
    print(hdr)
    sep = f"  {'-' * 4}  {'-' * 7}  {'-' * 9}  {'-' * 15}  {'-' * 6}  {'-' * 6}  {'-' * 15}  {'-' * 7}"
    print(sep)
    for trigger in [2, 3, 4, 5, 6, 7, 8]:
        result = run_backtest(candles, strategy, {"trigger": trigger, "size": 15.0})
        m = result.metrics
        n = m["trade_count"]
        wr = m["win_rate"]
        wins = round(wr * n)
        lo, hi = wilson_ci(wins, n)
        half_width = (hi - lo) / 2
        ci_str = f"[{lo:.1%}, {hi:.1%}]"

        est = get_rate_estimate(INTERVAL, trigger, asset)
        if est:
            poly_val = est.rate
            ci_str_poly = f"[{est.ci_lo:.1%},{est.ci_hi:.1%}]"
        else:
            tf_rates = REVERSAL_RATES.get(INTERVAL, REVERSAL_RATES["5m"])
            poly_val = tf_rates.get(trigger, tf_rates[max(tf_rates)])
            ci_str_poly = "n/a"

        poly_str = f"{poly_val:.1%}"
        delta = wr - poly_val
        delta_str = f"{delta:+.1%}"
        row = f"  {trigger:>4}  {n:>7}  {wr:>9.1%}  {ci_str:>15}"
        row += f"  {half_width:>5.1%}  {poly_str:>6}  {ci_str_poly:>15}  {delta_str:>7}"
        print(row)
    print()


def main() -> None:
    strategy = StreakReversalStrategy()

    # --- Fetch data ---
    now = datetime.now(tz=UTC)
    start = now - timedelta(days=LOOKBACK_DAYS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    asset = SYMBOL.replace("USDT", "").lower()
    parquet_path = Path("data") / f"{asset}_{INTERVAL}.parquet"
    if parquet_path.exists():
        print(f"Loading {parquet_path}...")
        raw = pd.read_parquet(parquet_path)
        raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
        cutoff = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        candles = raw[raw["open_time"] >= cutoff].set_index("open_time").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            candles[col] = pd.to_numeric(candles[col], errors="coerce")
    else:
        print(f"Fetching {SYMBOL} {INTERVAL} data ({LOOKBACK_DAYS} days)...")
        candles = _fetch_klines_vision(SYMBOL, INTERVAL, start_ms, end_ms)
        candles = candles.set_index("open_time").sort_index()
    print(f"  {len(candles):,} candles loaded  ({candles.index[0].date()} → {candles.index[-1].date()})\n")

    train, test = walk_forward_split(candles)
    print(f"Train: {len(train):,} candles | Test: {len(test):,} candles\n")

    # --- Default params ---
    print("=" * 60)
    print("DEFAULT PARAMS (trigger=4, size=15) — full dataset")
    print("=" * 60)
    default_result = run_backtest(candles, strategy)
    m = default_result.metrics
    print(f"  Trade count : {m['trade_count']}")
    print(f"  Win rate    : {m['win_rate']:.1%}")
    print(f"  Total PnL   : ${m['total_pnl']:+.2f}")
    print(f"  Max drawdown: ${m['max_drawdown']:.2f}")
    print(f"  Sharpe      : {m['sharpe_ratio']:.3f}")
    print()

    # --- Per-trigger breakdown vs REVERSAL_RATES ---
    win_rate_by_trigger(candles, strategy)

    # --- Param sweep on train ---
    print("=" * 60)
    print("PARAMETER SWEEP — train set (top 10 by win_rate)")
    print("=" * 60)
    param_grid = {"trigger": [2, 3, 4, 5, 6, 7, 8], "size": [10.0, 15.0, 20.0]}
    sweep = parameter_sweep(train, strategy, param_grid)
    top10 = sweep.head(10)
    print(top10[["trigger", "size", "win_rate", "total_pnl", "trade_count", "sharpe_ratio"]].to_string(index=False))
    print()

    # --- Best params on test set ---
    best_row = sweep.iloc[0].to_dict()
    best_params = {"trigger": int(best_row["trigger"]), "size": float(best_row["size"])}
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
    print("=" * 60)
    print("CONFIDENCE RECOMMENDATION")
    print("=" * 60)
    test_wr = m["win_rate"]
    test_count = m["trade_count"]
    if test_count < 30:
        print(f"  WARNING: only {test_count} test trades — sample too small for reliable estimate")
        print(f"  Fallback to REVERSAL_RATES table for trigger={best_params['trigger']}")
        tf_rates = REVERSAL_RATES.get(INTERVAL, REVERSAL_RATES["5m"])
        poly = tf_rates.get(best_params["trigger"], tf_rates[max(tf_rates)])
        print(f"  Polymarket historical rate: {poly:.1%}")
    else:
        print(f"  Measured test win rate : {test_wr:.3f}")
        print(f"  Suggested confidence   : {test_wr:.2f}  (use in streak_bot.py / REVERSAL_RATES)")
        if test_wr < 0.50:
            print("  NOTE: win rate < 50% — no edge at these params on test data")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=SYMBOL, help="Symbol (default: BTCUSDT)")
    parser.add_argument("--interval", default=INTERVAL, help="Candle interval (default: 5m)")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="Lookback days (default: 730)")
    args = parser.parse_args()
    SYMBOL = args.symbol
    INTERVAL = args.interval
    LOOKBACK_DAYS = args.days
    main()
