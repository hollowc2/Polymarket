#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich"]
# ///
"""Analyze signal_audit.jsonl to diagnose live bot signal gaps.

Usage:
    uv run --script scripts/analyze_signal_audit.py signal_audit.jsonl
    uv run --script scripts/analyze_signal_audit.py signal_audit.jsonl --timeline
"""

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime

from rich.console import Console  # type: ignore[import-untyped]
from rich.table import Table  # type: ignore[import-untyped]

console = Console()


def load_entries(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def fmt_ts(ts: int | None) -> str:
    if not ts:
        return "?"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%m-%d %H:%M")


def main():
    parser = argparse.ArgumentParser(description="Analyze signal_audit.jsonl")
    parser.add_argument("path", nargs="?", default="signal_audit.jsonl", help="Path to signal_audit.jsonl")
    parser.add_argument("--timeline", action="store_true", help="Print per-window timeline")
    parser.add_argument("--tail", type=int, default=0, metavar="N", help="Show only last N windows in timeline")
    args = parser.parse_args()

    try:
        entries = load_entries(args.path)
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print("No entries found.")
        return

    # ── Summary counts ────────────────────────────────────────────────────────
    reason_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    placed = 0
    filled = 0
    missed = 0

    cb_open_windows: list[dict] = []

    for e in entries:
        reason = e.get("reason", "unknown")
        outcome = e.get("outcome", "unknown")
        reason_counts[reason] += 1
        outcome_counts[outcome] += 1

        if reason == "circuit_open":
            cb_open_windows.append(e)
        if outcome == "placed":
            placed += 1
        elif outcome == "filled":
            filled += 1
        elif outcome == "missed_fill":
            missed += 1

    total = len(entries)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n=== Signal Audit Summary — {args.path} ===")
    print(f"Total entries : {total}")
    print(f"  placed      : {placed}")
    print(f"  filled      : {filled}")
    print(f"  missed_fill : {missed}")
    print(f"  skipped     : {outcome_counts.get('skipped', 0)}")
    print()

    # ── Breakdown by reason ───────────────────────────────────────────────────
    table = Table(title="Skip Reason Breakdown", show_lines=False)
    table.add_column("Reason", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("% of Total", justify="right")
    for reason, count in reason_counts.most_common():
        pct = 100 * count / total if total else 0
        color = "red" if reason in ("circuit_open", "rate_limited", "fetch_failed") else ""
        style = f"bold {color}" if color else ""
        table.add_row(reason, str(count), f"{pct:.1f}%", style=style)
    console.print(table)

    # ── Circuit breaker alert ─────────────────────────────────────────────────
    if cb_open_windows:
        print(f"\n[!] CIRCUIT BREAKER OPEN detected in {len(cb_open_windows)} windows:")
        for e in cb_open_windows[:10]:
            print(f"    {fmt_ts(e.get('ts'))}  {e.get('window_slug', '?')}")
        if len(cb_open_windows) > 10:
            print(f"    ... and {len(cb_open_windows) - 10} more")

    # ── Cancel failed alert ───────────────────────────────────────────────────
    cancel_failed_entries = [e for e in entries if e.get("reason") == "limit_expired" and e.get("cancel_failed")]
    if cancel_failed_entries:
        print(f"\n[!] CANCEL FAILED entries: {len(cancel_failed_entries)}")

    # ── Missed fill breakdown ─────────────────────────────────────────────────
    missed_reasons = Counter(e.get("reason") for e in entries if e.get("outcome") == "missed_fill")
    if missed_reasons:
        print("\nMissed fill sub-reasons:")
        for r, c in missed_reasons.most_common():
            print(f"  {r:<30} {c}")

    # ── Timeline ──────────────────────────────────────────────────────────────
    if args.timeline:
        print("\n── Window Timeline ──")
        timeline_entries = entries[-args.tail :] if args.tail else entries

        for e in timeline_entries:
            ts_str = fmt_ts(e.get("ts"))
            outcome = e.get("outcome", "?")
            reason = e.get("reason", "?")
            slug = e.get("window_slug") or "?"
            streak = e.get("streak_len", "?")
            ask = e.get("ask_price")
            ask_str = f"ask={ask:.4f}" if ask else ""
            limit = e.get("limit_price")
            limit_str = f"limit={limit:.4f}" if limit else ""

            if outcome == "filled":
                marker = "[+]"
            elif outcome == "placed":
                marker = "[->]"
            elif outcome == "missed_fill":
                marker = "[x]"
            else:
                marker = "[ ]"

            extras = " ".join(x for x in [ask_str, limit_str] if x)
            print(f"  {marker} {ts_str}  {slug:<45} {reason:<25} streak={streak}  {extras}")

    print()


if __name__ == "__main__":
    main()
