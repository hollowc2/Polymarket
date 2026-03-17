#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "pandas", "numpy", "pyarrow"]
# ///
"""Phase 2: Validate streak reversal signal against real Polymarket resolution data.

Samples timestamps where the streak signal fires, queries the Gamma API for
each market's actual resolution outcome, and reports whether the streak
reversal prediction matches real results.

Results are cached to disk — re-runs are instant.

Usage:
    uv run --script scripts/analyze_polymarket_signal.py
    uv run --script scripts/analyze_polymarket_signal.py --trigger 4 --samples 500
    uv run --script scripts/analyze_polymarket_signal.py --asset eth --tf 5m --trigger 3 4 5
    uv run --script scripts/analyze_polymarket_signal.py --no-cache   # force re-fetch
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = Path(__file__).resolve().parents[1] / "backtest_results" / "gamma_cache"
GAMMA_API = "https://gamma-api.polymarket.com"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_candles(asset: str, tf: str) -> pd.DataFrame:
    path = DATA_DIR / f"{asset}_{tf}.parquet"
    df = pd.read_parquet(path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
    return df.sort_index()


def compute_streak_signal(candles: pd.DataFrame, trigger: int) -> pd.Series:
    """Return signal series: -1 (bet DOWN), 0 (no trade), 1 (bet UP)."""
    direction = (candles["close"].diff() > 0).map({True: 1, False: -1}).fillna(0)
    streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
    signal = pd.Series(0, index=candles.index, dtype=int)
    signal[(streak >= trigger) & (direction == 1)] = -1  # streak UP → bet DOWN
    signal[(streak >= trigger) & (direction == -1)] = 1  # streak DOWN → bet UP
    return signal


# --------------------------------------------------------------------------- #
# Gamma API
# --------------------------------------------------------------------------- #


def fetch_market_outcome(session: requests.Session, asset: str, ts: int) -> str | None:
    """Query Gamma API for a single eth-updown-5m-{ts} market.

    Returns 'up', 'down', or None (market not found / not resolved).
    """
    slug = f"{asset}-updown-5m-{ts}"
    try:
        resp = session.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        markets = data[0].get("markets", [])
        if not markets:
            return None
        m = markets[0]
        prices = json.loads(m.get("outcomePrices", "[0.5,0.5]"))
        up_price = float(prices[0]) if prices else 0.5
        down_price = float(prices[1]) if len(prices) > 1 else 0.5
        is_closed = m.get("closed", False)
        uma_status = m.get("umaResolutionStatus", "")
        is_resolved = uma_status == "resolved" or up_price > 0.99 or down_price > 0.99
        if not (is_closed and is_resolved):
            return None  # still live or pending
        if up_price > 0.99:
            return "up"
        if down_price > 0.99:
            return "down"
        return None
    except Exception:
        return None


def load_cache(cache_path: Path) -> dict[str, str | None]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def save_cache(cache_path: Path, cache: dict[str, str | None]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def breakeven_win_rate(fill_price: float, fee_bps: int = 200) -> float:
    """Minimum win rate needed to break even at a given fill price."""
    fee = fee_bps / 10000
    win_payout = 1.0 - fee  # $0.98 net on a $1 win
    # E[PnL] = p * (win_payout - fill_price) - (1-p) * fill_price = 0
    # p = fill_price / win_payout
    return fill_price / win_payout


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate streak signal vs real Polymarket resolutions")
    parser.add_argument("--asset", default="eth")
    parser.add_argument("--tf", default="5m")
    parser.add_argument("--trigger", type=int, nargs="+", default=[3, 4, 5, 6], help="Streak trigger lengths to test")
    parser.add_argument("--samples", type=int, default=500, help="Max signal timestamps to query per trigger")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached Gamma API results and re-fetch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--cutoff", default="2024-01-01", help="Only use candles after this date (test period)")
    parser.add_argument(
        "--fill-price", type=float, default=0.51, help="Assumed fill price for break-even calc (default 0.51)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    candles = load_candles(args.asset, args.tf)

    # Restrict to test period (out-of-sample, avoids lookahead)
    cutoff_ts = pd.Timestamp(args.cutoff, tz="UTC")
    candles = candles.loc[candles.index >= cutoff_ts]
    print(f"Loaded {len(candles):,} candles ({args.asset}/{args.tf}) from {args.cutoff} onwards")

    # Cache keyed by asset+tf
    cache_path = CACHE_DIR / f"{args.asset}_{args.tf}_outcomes.json"
    cache: dict[str, str | None] = {} if args.no_cache else load_cache(cache_path)

    session = requests.Session()
    session.headers.update({"User-Agent": "PolymarketResearch/1.0", "Accept": "application/json"})

    # For each trigger length, compute signals and sample timestamps
    # We use trigger=min(triggers) to get ALL signal events, then filter per trigger
    min_trigger = min(args.trigger)

    # Compute direction and streak once
    direction = (candles["close"].diff() > 0).map({True: 1, False: -1}).fillna(0)
    streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1

    # The market we bet on is the NEXT candle's market.
    # Signal fires at candle T (streak visible after T closes), bet placed on market T+1.
    # Market slug uses T+1's open time = T's open time + 5min = T's timestamp + 300s.
    tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
    candle_seconds = tf_seconds.get(args.tf, 300)

    # Collect all timestamps where any trigger fires (for efficient batching)
    fires_at_min_trigger = (streak >= min_trigger) & (direction != 0)
    signal_candles = candles.loc[fires_at_min_trigger].copy()
    signal_candles["streak_len"] = streak.loc[fires_at_min_trigger]
    signal_candles["direction"] = direction.loc[fires_at_min_trigger]

    # Market timestamp = next candle open = this candle's open + candle_seconds
    # Index may be datetime64[ms] or datetime64[ns] depending on pandas version.
    # datetime64[ms] astype int64 → milliseconds; datetime64[ns] → nanoseconds.
    # Divide accordingly to get seconds.
    idx_int = signal_candles.index.astype("int64")
    divisor = 10**3 if "ms" in str(signal_candles.index.dtype) else 10**9
    signal_candles["market_ts"] = idx_int // divisor + candle_seconds

    # Sample across all triggers combined (avoid redundant API calls)
    all_ts = signal_candles["market_ts"].unique().tolist()
    sample_ts = random.sample(all_ts, min(args.samples * 2, len(all_ts)))  # oversample
    sample_set = set(sample_ts)

    # Filter signal_candles to only sampled timestamps
    signal_candles = signal_candles[signal_candles["market_ts"].isin(sample_set)]

    # Fetch outcomes for all sampled timestamps not already in cache
    to_fetch = [ts for ts in sample_ts if str(ts) not in cache]
    print(f"\nFetching {len(to_fetch)} markets from Gamma API (cached: {len(cache)})...")

    for i, ts in enumerate(to_fetch):
        outcome = fetch_market_outcome(session, args.asset, ts)
        cache[str(ts)] = outcome
        if (i + 1) % 50 == 0:
            save_cache(cache_path, cache)
            found = sum(1 for k in to_fetch[: i + 1] if cache.get(str(k)) is not None)
            print(f"  {i + 1}/{len(to_fetch)} fetched  ({found} resolved markets found)")
        time.sleep(0.15)  # gentle rate limit

    save_cache(cache_path, cache)
    print(f"Cache saved → {cache_path}")

    # ----------- Analysis per trigger length ----------- #
    print(f"\n{'=' * 75}")
    print(f"SIGNAL VALIDATION — {args.asset.upper()}/{args.tf} — test period from {args.cutoff}")
    print(f"Break-even at fill={args.fill_price:.2f}: {breakeven_win_rate(args.fill_price):.1%}")
    print(f"{'=' * 75}")
    header = f"{'Trigger':>8}  {'N':>5}  {'Resolved':>8}  {'Win%':>7}  {'95% CI':>14}  {'Edge?'}"
    print(header)
    print("-" * 75)

    for trig in sorted(args.trigger):
        # Filter to candles where this trigger fires
        trig_candles = signal_candles[signal_candles["streak_len"] >= trig]

        # Limit to args.samples
        if len(trig_candles) > args.samples:
            trig_candles = trig_candles.sample(args.samples, random_state=args.seed)

        resolved_rows = []
        for _, row in trig_candles.iterrows():
            ts_key = str(int(row["market_ts"]))
            outcome = cache.get(ts_key)
            if outcome is None:
                continue

            # Our signal: streak UP → bet DOWN, streak DOWN → bet UP
            signal_dir = "down" if row["direction"] == 1 else "up"
            is_win = signal_dir == outcome
            resolved_rows.append(is_win)

        n_resolved = len(resolved_rows)
        if n_resolved == 0:
            print(f"  trigger={trig}: no resolved markets found")
            continue

        wins = sum(resolved_rows)
        win_rate = wins / n_resolved
        lo, hi = wilson_ci(wins, n_resolved)
        be = breakeven_win_rate(args.fill_price)
        edge = "YES ✓" if lo > be else ("MAYBE" if win_rate > be else "NO ✗")

        print(f"  {trig:>6}  {len(trig_candles):>5}  {n_resolved:>8}  {win_rate:>7.1%}  [{lo:.1%} – {hi:.1%}]  {edge}")

    print("=" * 75)

    # ----------- By direction breakdown ----------- #
    print(f"\nBreakdown by streak direction (trigger={min_trigger}):")
    print(f"  {'Direction':>10}  {'N':>5}  {'Win%':>7}  {'95% CI':>14}")
    print(f"  {'-' * 45}")
    for trig in sorted(args.trigger)[:1]:  # just min trigger for brevity
        for streak_dir, label in [(1, "UP streak→DOWN bet"), (-1, "DOWN streak→UP bet")]:
            trig_candles = signal_candles[
                (signal_candles["streak_len"] >= trig) & (signal_candles["direction"] == streak_dir)
            ]
            resolved_rows = []
            for _, row in trig_candles.iterrows():
                ts_key = str(int(row["market_ts"]))
                outcome = cache.get(ts_key)
                if outcome is None:
                    continue
                signal_dir = "down" if row["direction"] == 1 else "up"
                resolved_rows.append(signal_dir == outcome)
            n = len(resolved_rows)
            if n == 0:
                continue
            w = sum(resolved_rows)
            lo, hi = wilson_ci(w, n)
            print(f"  {label:>20}  {n:>5}  {w / n:>7.1%}  [{lo:.1%} – {hi:.1%}]")

    # ----------- Month-by-month breakdown ----------- #
    print(f"\nMonth-by-month win rate (trigger={min_trigger}, any direction):")
    print(f"  {'Month':>8}  {'N':>4}  {'Win%':>7}  {'95% CI':>14}")
    print(f"  {'-' * 40}")
    trig_candles = signal_candles[signal_candles["streak_len"] >= min_trigger]
    monthly: dict[str, list[bool]] = {}
    for idx, row in trig_candles.iterrows():
        ts_key = str(int(row["market_ts"]))
        outcome = cache.get(ts_key)
        if outcome is None:
            continue
        month = str(idx)[:7]  # "2024-03"
        signal_dir = "down" if row["direction"] == 1 else "up"
        monthly.setdefault(month, []).append(signal_dir == outcome)
    for month in sorted(monthly):
        rows = monthly[month]
        n = len(rows)
        w = sum(rows)
        lo, hi = wilson_ci(w, n)
        bar = "█" * int((w / n) * 20) if n else ""
        print(f"  {month:>8}  {n:>4}  {w / n:>7.1%}  [{lo:.1%} – {hi:.1%}]  {bar}")

    print()
    total_resolved = sum(1 for v in cache.values() if v is not None)
    total_queried = len(cache)
    hit_rate = total_resolved / total_queried if total_queried else 0
    print(f"Cache: {total_queried} timestamps queried, {total_resolved} resolved ({hit_rate:.0%} hit rate)")


if __name__ == "__main__":
    main()
