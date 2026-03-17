#!/usr/bin/env python3
"""Analyze actual fill prices from trade_history_full.json.

Answers: "What are we actually paying vs the 0.50 baseline?"

Note: streak_length is not stored in the JSON history (it's in-memory only).
Instead, groups trades by best_ask price bucket at entry (a proxy for market
sentiment after a streak) and also shows overall fill price stats.

Usage:
    uv run python scripts/analyze_fills.py
    uv run python scripts/analyze_fills.py trade_history_full.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("trade_history_full.json")
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with path.open() as f:
        data = json.load(f)

    if not data:
        print("No trades in history file.")
        return

    settled = [t for t in data if t.get("settlement", {}).get("status") == "settled"]
    pending = [t for t in data if t.get("settlement", {}).get("status") == "pending"]

    print(f"\nLoaded {len(data)} total trades ({len(settled)} settled, {len(pending)} pending)\n")

    # ── Per-direction fill prices ─────────────────────────────────────────────
    by_direction: dict[str, list[float]] = defaultdict(list)
    entry_prices: list[float] = []
    fill_prices: list[float] = []
    ask_at_entries: list[float] = []

    for t in settled:
        execution = t.get("execution", {})
        position = t.get("position", {})
        direction = position.get("direction", "?")
        entry = execution.get("entry_price", 0.0)
        fill = execution.get("fill_price", 0.0)
        ask = execution.get("best_ask", 0.0)

        if fill > 0:
            by_direction[direction].append(fill)
            fill_prices.append(fill)
        if entry > 0:
            entry_prices.append(entry)
        if ask > 0:
            ask_at_entries.append(ask)

    # ── By direction ──────────────────────────────────────────────────────────
    print("Fill prices by direction (vs 0.50 baseline):")
    print(f"  {'Direction':<10} {'Count':>5} {'Avg Entry':>10} {'Avg Fill':>10} {'vs 0.50':>8}")
    print(f"  {'-' * 10} {'-' * 5} {'-' * 10} {'-' * 10} {'-' * 8}")
    for direction in ("up", "down"):
        fills = by_direction.get(direction, [])
        if not fills:
            print(f"  {direction:<10} {'0':>5}")
            continue
        avg_fill = sum(fills) / len(fills)
        # For entry_price, gather just this direction
        dir_entries = [
            t.get("execution", {}).get("entry_price", 0.0)
            for t in settled
            if t.get("position", {}).get("direction") == direction
        ]
        avg_entry = sum(e for e in dir_entries if e > 0) / max(1, len([e for e in dir_entries if e > 0]))
        delta = avg_fill - 0.50
        sign = "+" if delta >= 0 else ""
        print(f"  {direction:<10} {len(fills):>5} {avg_entry:>10.4f} {avg_fill:>10.4f} {sign}{delta:>7.4f}")

    # ── Overall ───────────────────────────────────────────────────────────────
    if fill_prices:
        avg_fill_overall = sum(fill_prices) / len(fill_prices)
        avg_entry_overall = sum(entry_prices) / len(entry_prices) if entry_prices else 0.0
        delta_overall = avg_fill_overall - 0.50
        sign = "+" if delta_overall >= 0 else ""
        overall_line = (
            f"\n  {'OVERALL':<10} {len(fill_prices):>5} "
            f"{avg_entry_overall:>10.4f} {avg_fill_overall:>10.4f} {sign}{delta_overall:>7.4f}"
        )
        print(overall_line)

    # ── Fill price buckets (proxy for market sentiment) ────────────────────────
    # Group by best_ask at entry in 0.05 buckets
    print("\n\nFill prices by best_ask at entry (proxy for market sentiment/streak):")
    print("  (Lower ask = crowd expects DOWN more strongly → better reversal opportunity)")
    print(f"\n  {'Ask bucket':<12} {'Count':>5} {'Avg Fill':>10} {'vs 0.50':>8} {'Win rate':>9}")
    print(f"  {'-' * 12} {'-' * 5} {'-' * 10} {'-' * 8} {'-' * 9}")

    buckets: dict[str, list] = defaultdict(list)
    for t in settled:
        execution = t.get("execution", {})
        position = t.get("position", {})
        settlement = t.get("settlement", {})
        ask = execution.get("best_ask", 0.0)
        fill = execution.get("fill_price", 0.0)
        won = settlement.get("won", None)

        if ask <= 0 or fill <= 0:
            continue

        # Bucket by 0.05 increments
        bucket_low = round(int(ask * 20) / 20, 2)
        bucket_high = round(bucket_low + 0.05, 2)
        key = f"{bucket_low:.2f}-{bucket_high:.2f}"
        buckets[key].append((fill, won))

    for key in sorted(buckets.keys()):
        items = buckets[key]
        fills = [x[0] for x in items]
        wins = [x[1] for x in items if x[1] is not None]
        avg_fill = sum(fills) / len(fills)
        win_rate = (sum(1 for w in wins if w) / len(wins) * 100) if wins else 0.0
        delta = avg_fill - 0.50
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<12} {len(fills):>5} {avg_fill:>10.4f} {sign}{delta:>7.4f} {win_rate:>8.1f}%")

    # ── Slippage summary ──────────────────────────────────────────────────────
    slippages = [
        t.get("execution", {}).get("slippage_pct", 0.0)
        for t in settled
        if t.get("execution", {}).get("slippage_pct", 0.0) > 0
    ]
    spreads = [
        t.get("execution", {}).get("spread", 0.0) for t in settled if t.get("execution", {}).get("spread", 0.0) > 0
    ]

    print("\n\nExecution quality summary (settled trades):")
    if slippages:
        print(f"  Avg slippage : {sum(slippages) / len(slippages):.4f}%  (n={len(slippages)})")
        print(f"  Max slippage : {max(slippages):.4f}%")
    if spreads:
        print(f"  Avg spread   : {sum(spreads) / len(spreads) * 100:.2f}¢")
    if ask_at_entries and fill_prices:
        # How much above the best_ask did we actually pay on average?
        overpay = []
        for t in settled:
            ex = t.get("execution", {})
            ask = ex.get("best_ask", 0.0)
            fill = ex.get("fill_price", 0.0)
            if ask > 0 and fill > 0:
                overpay.append(fill - ask)
        if overpay:
            avg_overpay = sum(overpay) / len(overpay)
            sign = "+" if avg_overpay >= 0 else ""
            print(f"  Avg overpay  : {sign}{avg_overpay * 100:.2f}¢ above best_ask")

    # ── EV impact of limit orders ─────────────────────────────────────────────
    print("\n\nEstimated EV impact of 3¢ limit discount:")
    if fill_prices:
        baseline_ev = avg_fill_overall - 0.50 if fill_prices else 0.0
        limit_fill = avg_fill_overall - 0.03  # rough estimate: discount saves ~3¢
        limit_ev = limit_fill - 0.50
        print(f"  Current avg fill  : {avg_fill_overall:.4f}  (EV vs 0.50 = {baseline_ev:+.4f})")
        print(f"  With 3¢ discount  : {limit_fill:.4f}  (EV vs 0.50 = {limit_ev:+.4f})")
        pct_improvement = abs((limit_ev - baseline_ev) / baseline_ev * 100) if baseline_ev != 0 else float("inf")
        print(f"  EV improvement    : {pct_improvement:.0f}% (if limit fills)")

    print()


if __name__ == "__main__":
    main()
