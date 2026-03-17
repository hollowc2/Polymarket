#!/usr/bin/env python3
"""Compare two bot history files side-by-side.

Useful for A/B testing different entry strategies running simultaneously
(e.g. streak-bot vs alt-entry-bot) against the same markets.

Usage:
    uv run python scripts/compare_bots.py \\
        --a  state/trade_history_full.json \\
        --b  state/alt-entry-history.json \\
        --label-a "streak-bot (market)" \\
        --label-b "alt-entry (spread/mid)"

    # Quick default (looks for the two VPS state filenames locally):
    uv run python scripts/compare_bots.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── ANSI ────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def g(s: str) -> str:
    return f"{GREEN}{s}{RESET}"


def r(s: str) -> str:
    return f"{RED}{s}{RESET}"


def y(s: str) -> str:
    return f"{YELLOW}{s}{RESET}"


def b(s: str) -> str:
    return f"{BOLD}{s}{RESET}"


def d(s: str) -> str:
    return f"{DIM}{s}{RESET}"


def color_val(val: float, good_positive: bool = True) -> str:
    if val > 0:
        return g(f"{val:+.2f}") if good_positive else r(f"{val:+.2f}")
    elif val < 0:
        return r(f"{val:+.2f}") if good_positive else g(f"{val:+.2f}")
    return y(f"{val:+.2f}")


def color_pct(pct: float, good_positive: bool = True) -> str:
    fmt = f"{pct:+.1f}%"
    if pct > 0:
        return g(fmt) if good_positive else r(fmt)
    elif pct < 0:
        return r(fmt) if good_positive else g(fmt)
    return y(fmt)


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    slug: str
    direction: str
    amount: float
    best_ask: float
    best_bid: float
    spread: float
    fill_price: float
    hour_utc: int
    won: bool
    net_profit: float
    order_type: str  # "market" or "limit"
    limit_price: float | None
    missed: bool  # limit order that expired unfilled
    mode: str


def load_history(path: str) -> list[TradeRecord]:
    with open(path) as f:
        raw: list[dict] = json.load(f)

    records: list[TradeRecord] = []
    for item in raw:
        try:
            exec_ = item.get("execution", {})
            settlement = item.get("settlement", {})
            timing = item.get("timing", {})
            position = item.get("position", {})
            context = item.get("context", {})
            limit_info = item.get("limit", {})

            if settlement.get("status") not in ("settled", "forced_exit"):
                continue
            if settlement.get("won") is None:
                continue

            best_ask = float(exec_.get("best_ask", 0.0))
            best_bid = float(exec_.get("best_bid", 0.0))
            spread = float(exec_.get("spread", best_ask - best_bid))

            records.append(
                TradeRecord(
                    slug=item.get("market", {}).get("slug", ""),
                    direction=position.get("direction", ""),
                    amount=float(position.get("amount", 0)),
                    best_ask=best_ask,
                    best_bid=best_bid,
                    spread=spread,
                    fill_price=float(exec_.get("fill_price", exec_.get("entry_price", 0.5))),
                    hour_utc=int(timing.get("hour_utc", 0)),
                    won=bool(settlement.get("won", False)),
                    net_profit=float(settlement.get("net_profit", 0.0)),
                    order_type=limit_info.get("order_type", "market"),
                    limit_price=limit_info.get("limit_price"),
                    missed=bool(limit_info.get("missed", False)),
                    mode=context.get("mode", "paper"),
                )
            )
        except Exception:
            continue

    return records


# ── Stats ─────────────────────────────────────────────────────────────────────


@dataclass
class Stats:
    label: str
    trades: list[TradeRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        return sum(1 for t in self.trades if t.won) / self.n if self.n else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.net_profit for t in self.trades)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.n if self.n else 0.0

    @property
    def avg_entry(self) -> float:
        valid = [t.fill_price for t in self.trades if t.fill_price > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_ask(self) -> float:
        valid = [t.best_ask for t in self.trades if t.best_ask > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_spread(self) -> float:
        valid = [t.spread for t in self.trades if t.spread > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_discount(self) -> float:
        """Average (ask - fill_price) — how much cheaper than ask we got."""
        vals = [t.best_ask - t.fill_price for t in self.trades if t.best_ask > 0 and t.fill_price > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def breakeven_wr(self) -> float:
        p = self.avg_entry
        if p <= 0:
            return 1.0
        fee = p * (1 - p) * 0.10
        return min(1.0, p + fee)

    @property
    def edge(self) -> float:
        return self.win_rate - self.breakeven_wr

    @property
    def limit_fill_rate(self) -> float | None:
        limits = [t for t in self.trades if t.order_type == "limit"]
        if not limits:
            return None
        filled = sum(1 for t in limits if not t.missed)
        return filled / len(limits)


def find_common_markets(a: list[TradeRecord], b: list[TradeRecord]) -> tuple[list[TradeRecord], list[TradeRecord]]:
    """Return only trades that targeted the same market slug in both histories."""
    slugs_a = {t.slug for t in a}
    slugs_b = {t.slug for t in b}
    common = slugs_a & slugs_b
    return (
        [t for t in a if t.slug in common],
        [t for t in b if t.slug in common],
    )


# ── Display ───────────────────────────────────────────────────────────────────


COL = 28  # column width for each bot


def _strip(s: str) -> str:
    for code in (GREEN, RED, YELLOW, CYAN, BOLD, DIM, RESET):
        s = s.replace(code, "")
    return s


def row(label: str, val_a: str, val_b: str, delta: str = "") -> None:
    clean_a = _strip(val_a)
    clean_b = _strip(val_b)
    pad_a = COL - len(clean_a)
    pad_b = COL - len(clean_b)
    print(f"  {label:<26}{val_a}{' ' * pad_a}{val_b}{' ' * pad_b}{delta}")


def section(title: str) -> None:
    print(b(f"\n{'─' * 70}"))
    print(b(f"  {title}"))
    print(b(f"{'─' * 70}"))


def print_comparison(sa: Stats, sb: Stats, common_only: bool) -> None:
    qualifier = " (common markets only)" if common_only else " (all settled trades)"

    print(b(f"\n{'=' * 70}"))
    print(b(f"  BOT COMPARISON{qualifier}"))
    print(b(f"{'=' * 70}"))
    print(f"  {'':26}{b(sa.label[:COL]):<{COL + 11}}{b(sb.label[:COL])}")

    section("VOLUME")
    row("Trades", str(sa.n), str(sb.n))

    section("PERFORMANCE")
    row(
        "Win rate",
        f"{sa.win_rate:.1%}",
        f"{sb.win_rate:.1%}",
        color_pct((sb.win_rate - sa.win_rate) * 100, good_positive=True) + "  (B vs A)",
    )
    row("Break-even WR", f"{sa.breakeven_wr:.1%}", f"{sb.breakeven_wr:.1%}")
    row(
        "Edge (WR - BE)",
        color_pct(sa.edge * 100),
        color_pct(sb.edge * 100),
        color_pct((sb.edge - sa.edge) * 100) + "  (B vs A)",
    )
    row(
        "Total PnL",
        color_val(sa.total_pnl),
        color_val(sb.total_pnl),
        color_val(sb.total_pnl - sa.total_pnl) + "  (B vs A)",
    )
    row(
        "Avg PnL / trade",
        color_val(sa.avg_pnl),
        color_val(sb.avg_pnl),
        color_val(sb.avg_pnl - sa.avg_pnl) + "  (B vs A)",
    )

    section("ENTRY QUALITY")
    row(
        "Avg fill price",
        f"{sa.avg_entry:.4f}" if sa.avg_entry else "-",
        f"{sb.avg_entry:.4f}" if sb.avg_entry else "-",
        (color_val(sa.avg_entry - sb.avg_entry, good_positive=True) + "  (A cheaper)")
        if sa.avg_entry and sb.avg_entry
        else "",
    )
    row("Avg best ask", f"{sa.avg_ask:.4f}" if sa.avg_ask else "-", f"{sb.avg_ask:.4f}" if sb.avg_ask else "-")
    row(
        "Avg discount vs ask",
        f"{sa.avg_discount:.4f}" if sa.avg_ask else "-",
        f"{sb.avg_discount:.4f}" if sb.avg_ask else "-",
        (color_val(sb.avg_discount - sa.avg_discount, good_positive=True) + "  (B better)")
        if sa.avg_ask and sb.avg_ask
        else "",
    )
    row(
        "Avg spread", f"{sa.avg_spread:.4f}" if sa.avg_spread else "-", f"{sb.avg_spread:.4f}" if sb.avg_spread else "-"
    )

    fill_a = sa.limit_fill_rate
    fill_b = sb.limit_fill_rate
    if fill_a is not None or fill_b is not None:
        section("LIMIT ORDER FILL RATE")
        row(
            "Limit fill rate",
            f"{fill_a:.1%}" if fill_a is not None else d("N/A (market orders)"),
            f"{fill_b:.1%}" if fill_b is not None else d("N/A (market orders)"),
        )

    print()


def print_per_hour(sa: Stats, sb: Stats) -> None:
    all_hours = sorted({t.hour_utc for t in sa.trades} | {t.hour_utc for t in sb.trades})
    if not all_hours:
        return

    section("WIN RATE BY UTC HOUR")
    print(f"  {'Hour':10}{'A win%':>10}  {'A n':>5}    {'B win%':>10}  {'B n':>5}    {'Δ win%':>10}")
    print(d("  " + "-" * 60))
    for h in all_hours:
        ta = [t for t in sa.trades if t.hour_utc == h]
        tb = [t for t in sb.trades if t.hour_utc == h]
        wr_a = sum(1 for t in ta if t.won) / len(ta) if ta else None
        wr_b = sum(1 for t in tb if t.won) / len(tb) if tb else None
        wr_a_str = f"{wr_a:.1%}" if wr_a is not None else "-"
        wr_b_str = f"{wr_b:.1%}" if wr_b is not None else "-"
        na_str = str(len(ta)) if ta else "-"
        nb_str = str(len(tb)) if tb else "-"
        if wr_a is not None and wr_b is not None:
            delta = color_pct((wr_b - wr_a) * 100)
        else:
            delta = ""
        print(f"  {h:02d}:00     {wr_a_str:>10}  {na_str:>5}    {wr_b_str:>10}  {nb_str:>5}    {delta}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two bot history files side-by-side",
    )
    parser.add_argument(
        "--a",
        default="trade_history_full.json",
        metavar="PATH",
        help="History file for bot A (default: trade_history_full.json)",
    )
    parser.add_argument(
        "--b",
        default="alt-entry-history.json",
        metavar="PATH",
        help="History file for bot B (default: alt-entry-history.json)",
    )
    parser.add_argument(
        "--label-a",
        default="streak-bot (market)",
        metavar="LABEL",
        help="Display name for bot A",
    )
    parser.add_argument(
        "--label-b",
        default="alt-entry (spread/mid)",
        metavar="LABEL",
        help="Display name for bot B",
    )
    parser.add_argument(
        "--common-only",
        action="store_true",
        default=False,
        help="Only compare trades on markets that both bots traded (fairer if different fire rates)",
    )
    parser.add_argument(
        "--by-hour",
        action="store_true",
        default=False,
        help="Also print win-rate breakdown by UTC hour",
    )
    args = parser.parse_args()

    missing = [p for p in [args.a, args.b] if not Path(p).exists()]
    if missing:
        for p in missing:
            print(r(f"Error: file not found: {p}"))
        print(d("Run both bots for a while to accumulate history, then re-run this script."))
        sys.exit(1)

    print(b("\nLoading histories..."))
    trades_a = load_history(args.a)
    trades_b = load_history(args.b)
    print(f"  {args.label_a}: {len(trades_a)} settled trades")
    print(f"  {args.label_b}: {len(trades_b)} settled trades")

    if args.common_only:
        trades_a, trades_b = find_common_markets(trades_a, trades_b)
        print(d(f"  After common-market filter: A={len(trades_a)}, B={len(trades_b)}"))

    sa = Stats(label=args.label_a, trades=trades_a)
    sb = Stats(label=args.label_b, trades=trades_b)

    if sa.n == 0 and sb.n == 0:
        print(r("No settled trades found in either file. Run the bots longer before comparing."))
        sys.exit(0)

    print_comparison(sa, sb, common_only=args.common_only)

    if args.by_hour:
        print_per_hour(sa, sb)


if __name__ == "__main__":
    main()
