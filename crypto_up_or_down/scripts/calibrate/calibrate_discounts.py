#!/usr/bin/env python3
"""Calibrate limit-order discount settings from historical fill data.

Reads trade_history_full.json and produces:
  1. Spread-width analysis — win rate and EV per narrow/medium/wide bracket
  2. Ask-price bracket analysis — which entry price ranges have the best edge
  3. UTC hour heatmap — when is edge highest
  4. Mid-price simulation — what would PnL look like if we had entered at mid
  5. Recommendation — suggested ALT_ENTRY_DISCOUNT_FRACTION for each bracket

Usage:
    uv run python scripts/calibrate_discounts.py
    uv run python scripts/calibrate_discounts.py --history path/to/trade_history_full.json
    uv run python scripts/calibrate_discounts.py --min-trades 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── ANSI colours ────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def green(s: str) -> str:
    return f"{GREEN}{s}{RESET}"


def red(s: str) -> str:
    return f"{RED}{s}{RESET}"


def yellow(s: str) -> str:
    return f"{YELLOW}{s}{RESET}"


def cyan(s: str) -> str:
    return f"{CYAN}{s}{RESET}"


def bold(s: str) -> str:
    return f"{BOLD}{s}{RESET}"


def dim(s: str) -> str:
    return f"{DIM}{s}{RESET}"


def color_pct(pct: float, threshold: float = 0.0) -> str:
    fmt = f"{pct:+.1f}%"
    if pct > threshold:
        return green(fmt)
    elif pct < -threshold:
        return red(fmt)
    return yellow(fmt)


# ── Data loading ────────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    slug: str
    direction: str
    amount: float
    entry_price: float
    fill_price: float
    best_bid: float
    best_ask: float
    spread: float
    slippage_pct: float
    hour_utc: int
    won: bool
    net_profit: float
    fee_pct: float
    mode: str  # "paper" or "live"


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

            # Skip unsettled trades
            if settlement.get("status") not in ("settled", "forced_exit"):
                continue
            if settlement.get("won") is None:
                continue

            best_ask = exec_.get("best_ask", 0.0)
            best_bid = exec_.get("best_bid", 0.0)
            spread = exec_.get("spread", best_ask - best_bid)
            fee_pct = item.get("fees", {}).get("pct", 0.025)

            records.append(
                TradeRecord(
                    slug=item.get("market", {}).get("slug", ""),
                    direction=position.get("direction", ""),
                    amount=float(position.get("amount", 0)),
                    entry_price=float(exec_.get("entry_price", 0.5)),
                    fill_price=float(exec_.get("fill_price", exec_.get("entry_price", 0.5))),
                    best_bid=float(best_bid),
                    best_ask=float(best_ask),
                    spread=float(spread),
                    slippage_pct=float(exec_.get("slippage_pct", 0.0)),
                    hour_utc=int(timing.get("hour_utc", 0)),
                    won=bool(settlement.get("won", False)),
                    net_profit=float(settlement.get("net_profit", 0.0)),
                    fee_pct=float(fee_pct),
                    mode=context.get("mode", "paper"),
                )
            )
        except Exception:
            continue

    return records


# ── Statistics helpers ───────────────────────────────────────────────────────


@dataclass
class GroupStats:
    label: str
    trades: list[TradeRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.won) / len(self.trades)

    @property
    def avg_net_profit(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.net_profit for t in self.trades) / len(self.trades)

    @property
    def total_net_profit(self) -> float:
        return sum(t.net_profit for t in self.trades)

    @property
    def avg_ask(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.best_ask for t in self.trades) / len(self.trades)

    @property
    def avg_spread(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.spread for t in self.trades) / len(self.trades)

    @property
    def avg_fee_pct(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.fee_pct for t in self.trades) / len(self.trades)

    def breakeven_win_rate(self, entry_price: float) -> float:
        """Minimum win rate to break even at given entry price (with fees).

        Simplified: you buy at P, pay fee ≈ P*(1-P)*fee_bps/10000.
        If you win you get $1 per share. Shares = amount/P.
        Break-even: P + fee = win_rate * 1.0  →  win_rate = P + fee
        """
        if entry_price <= 0:
            return 1.0
        fee = entry_price * (1 - entry_price) * 0.10  # approx 10% base fee
        return min(1.0, entry_price + fee)

    def mid_price_simulation(self) -> tuple[float, float, float]:
        """Simulate PnL if each trade had been placed at mid-price.

        Returns: (actual_total_pnl, simulated_mid_total_pnl, improvement)
        """
        actual = 0.0
        simulated = 0.0
        for t in self.trades:
            actual += t.net_profit
            if t.best_ask <= 0 or t.best_bid <= 0:
                simulated += t.net_profit
                continue
            mid = (t.best_ask + t.best_bid) / 2
            if mid <= 0:
                simulated += t.net_profit
                continue
            # At mid price: buy more shares for the same USD
            shares_at_mid = t.amount / mid
            fee_at_mid = mid * (1 - mid) * t.fee_pct  # fee_pct already as fraction
            if t.won:
                gross = shares_at_mid * 1.0
                fee_usd = shares_at_mid * fee_at_mid
                sim_net = gross - t.amount - fee_usd
            else:
                sim_net = -t.amount
            simulated += sim_net
        return actual, simulated, simulated - actual


# ── Grouping helpers ─────────────────────────────────────────────────────────


SPREAD_BRACKETS = [
    ("narrow  (<1¢)", 0.0, 0.01),
    ("medium (1–3¢)", 0.01, 0.03),
    ("wide    (>3¢)", 0.03, 999.0),
]

ASK_BRACKETS = [
    ("ask < 0.40", 0.0, 0.40),
    ("0.40–0.43", 0.40, 0.43),
    ("0.43–0.46", 0.43, 0.46),
    ("ask > 0.46", 0.46, 999.0),
]


def group_by_spread(records: list[TradeRecord]) -> list[GroupStats]:
    groups: list[GroupStats] = [GroupStats(label=label) for label, _, _ in SPREAD_BRACKETS]
    for rec in records:
        for i, (_, lo, hi) in enumerate(SPREAD_BRACKETS):
            if lo <= rec.spread < hi:
                groups[i].trades.append(rec)
                break
    return groups


def group_by_ask(records: list[TradeRecord]) -> list[GroupStats]:
    groups: list[GroupStats] = [GroupStats(label=label) for label, _, _ in ASK_BRACKETS]
    for rec in records:
        for i, (_, lo, hi) in enumerate(ASK_BRACKETS):
            if lo <= rec.best_ask < hi:
                groups[i].trades.append(rec)
                break
    return groups


def group_by_hour(records: list[TradeRecord]) -> dict[int, GroupStats]:
    by_hour: dict[int, GroupStats] = {}
    for rec in records:
        h = rec.hour_utc
        if h not in by_hour:
            by_hour[h] = GroupStats(label=f"UTC {h:02d}:00")
        by_hour[h].trades.append(rec)
    return dict(sorted(by_hour.items()))


# ── Recommendation logic ──────────────────────────────────────────────────────


def suggest_fraction(
    win_rate: float,
    avg_ask: float,
    avg_spread: float,
) -> str:
    """Suggest a discount fraction given observed win rate and market conditions."""
    if avg_spread <= 0:
        return "N/A (no spread data)"

    # Break-even WR at various fractions
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_frac = None
    best_margin = -999.0

    for f in fractions:
        price = avg_ask - avg_spread * f
        if price <= 0:
            continue
        be = price + price * (1 - price) * 0.10
        margin = win_rate - be
        if margin > best_margin:
            best_margin = margin
            best_frac = f

    if best_frac is None:
        return "insufficient data"
    if best_margin <= 0:
        return f"{best_frac:.1f} (warning: WR still below breakeven at all tested fractions)"
    return f"{best_frac:.1f}  (margin={best_margin:+.2%} vs breakeven)"


# ── Display helpers ───────────────────────────────────────────────────────────


def print_table(
    headers: list[str],
    rows: list[list[str]],
    min_width: int = 8,
) -> None:
    col_widths = [max(min_width, len(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            # strip ANSI when measuring
            clean = (
                cell.replace(GREEN, "")
                .replace(RED, "")
                .replace(YELLOW, "")
                .replace(CYAN, "")
                .replace(BOLD, "")
                .replace(DIM, "")
                .replace(RESET, "")
            )
            col_widths[i] = max(col_widths[i], len(clean))

    sep = "  "
    header_line = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(bold(header_line))
    print(dim("-" * len(header_line)))
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            clean = (
                cell.replace(GREEN, "")
                .replace(RED, "")
                .replace(YELLOW, "")
                .replace(CYAN, "")
                .replace(BOLD, "")
                .replace(DIM, "")
                .replace(RESET, "")
            )
            padding = col_widths[i] - len(clean)
            cells.append(cell + " " * padding)
        print(sep.join(cells))


# ── Section printers ─────────────────────────────────────────────────────────


def print_spread_analysis(groups: list[GroupStats], min_trades: int) -> None:
    print(bold(f"\n{'=' * 60}"))
    print(bold("  SPREAD-WIDTH ANALYSIS"))
    print(bold(f"{'=' * 60}"))
    print(dim("  Groups trades by bid-ask spread width at entry."))
    print(dim("  'Mid sim' = estimated PnL if you had entered at mid-price.\n"))

    headers = ["Bracket", "N", "Win%", "Avg ask", "Avg spread", "Avg PnL/trade", "Mid sim ΔPnL", "Suggest fraction"]
    rows = []
    for g in groups:
        if g.n < min_trades:
            rows.append(
                [
                    g.label,
                    str(g.n),
                    dim("(too few)"),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                ]
            )
            continue
        actual, simulated, improvement = g.mid_price_simulation()
        per_trade_improvement = improvement / g.n if g.n > 0 else 0
        frac = suggest_fraction(g.win_rate, g.avg_ask, g.avg_spread)
        rows.append(
            [
                g.label,
                str(g.n),
                f"{g.win_rate:.1%}",
                f"{g.avg_ask:.3f}",
                f"{g.avg_spread:.4f}",
                f"${g.avg_net_profit:+.2f}",
                f"${per_trade_improvement:+.2f}",
                frac,
            ]
        )

    print_table(headers, rows)


def print_ask_analysis(groups: list[GroupStats], min_trades: int) -> None:
    print(bold(f"\n{'=' * 60}"))
    print(bold("  ASK-PRICE BRACKET ANALYSIS"))
    print(bold(f"{'=' * 60}"))
    print(dim("  Groups trades by best_ask at entry. Lower ask = cheaper reversal entry.\n"))

    headers = ["Bracket", "N", "Win%", "BE win%", "Margin", "Total PnL", "Avg spread"]
    rows = []
    for g in groups:
        if g.n < min_trades:
            rows.append([g.label, str(g.n), dim("(too few)"), "-", "-", "-", "-"])
            continue
        be = g.breakeven_win_rate(g.avg_ask)
        margin = g.win_rate - be
        rows.append(
            [
                g.label,
                str(g.n),
                f"{g.win_rate:.1%}",
                f"{be:.1%}",
                color_pct(margin * 100, threshold=1.0),
                color_pct(g.total_net_profit),
                f"{g.avg_spread:.4f}",
            ]
        )

    print_table(headers, rows)


def print_hour_analysis(by_hour: dict[int, GroupStats], min_trades: int) -> None:
    print(bold(f"\n{'=' * 60}"))
    print(bold("  UTC HOUR HEATMAP"))
    print(bold(f"{'=' * 60}"))
    print(dim("  Trade activity and win rate by UTC hour.\n"))

    headers = ["Hour (UTC)", "N", "Win%", "Avg PnL/trade", "Total PnL"]
    rows = []
    for h, g in by_hour.items():
        if g.n < min_trades:
            rows.append([f"{h:02d}:00", str(g.n), dim("(too few)"), "-", "-"])
            continue
        rows.append(
            [
                f"{h:02d}:00",
                str(g.n),
                f"{g.win_rate:.1%}",
                f"${g.avg_net_profit:+.2f}",
                color_pct(g.total_net_profit),
            ]
        )

    print_table(headers, rows)


def print_mid_simulation(records: list[TradeRecord]) -> None:
    print(bold(f"\n{'=' * 60}"))
    print(bold("  MID-PRICE SIMULATION (overall)"))
    print(bold(f"{'=' * 60}"))
    print(dim("  Estimates total PnL if every trade had been placed at mid.\n"))

    total_actual = sum(r.net_profit for r in records)
    total_sim = 0.0
    for r in records:
        if r.best_ask <= 0 or r.best_bid <= 0:
            total_sim += r.net_profit
            continue
        mid = (r.best_ask + r.best_bid) / 2
        if mid <= 0:
            total_sim += r.net_profit
            continue
        shares_at_mid = r.amount / mid
        fee_at_mid = mid * (1 - mid) * r.fee_pct
        if r.won:
            gross = shares_at_mid * 1.0
            fee_usd = shares_at_mid * fee_at_mid
            total_sim += gross - r.amount - fee_usd
        else:
            total_sim += -r.amount

    n = len(records)
    improvement = total_sim - total_actual
    per_trade = improvement / n if n > 0 else 0

    print(f"  Trades analysed      : {n}")
    print(f"  Actual total PnL     : {color_pct(total_actual)}")
    print(f"  Simulated (at mid)   : {color_pct(total_sim)}")
    print(f"  Improvement          : {color_pct(improvement)}  ({color_pct(per_trade)} / trade)")
    if improvement > 0:
        print(
            f"\n  {green('► Mid-price entry would have improved total PnL by')} "
            f"{green(f'${improvement:+.2f}')} "
            f"{green(f'(${per_trade:+.2f}/trade)')}"
        )
    else:
        print(f"\n  {yellow('► No improvement from mid-price targeting in this dataset.')}")


def print_recommendations(records: list[TradeRecord], min_trades: int) -> None:
    print(bold(f"\n{'=' * 60}"))
    print(bold("  RECOMMENDATIONS"))
    print(bold(f"{'=' * 60}"))

    # Overall
    n = len(records)
    overall_wr = sum(1 for r in records if r.won) / n if n else 0
    avg_ask = sum(r.best_ask for r in records) / n if n else 0
    avg_spread = sum(r.spread for r in records) / n if n else 0

    print(
        f"\n  Overall: {n} settled trades | win rate {overall_wr:.1%} | "
        f"avg ask {avg_ask:.3f} | avg spread {avg_spread:.4f}"
    )

    # Suggested config
    frac = suggest_fraction(overall_wr, avg_ask, avg_spread)
    be_at_ask = avg_ask + avg_ask * (1 - avg_ask) * 0.10
    be_at_mid = (avg_ask - avg_spread * 0.5) + (avg_ask - avg_spread * 0.5) * (1 - (avg_ask - avg_spread * 0.5)) * 0.10

    print(f"\n  Break-even WR @ ask  : {be_at_ask:.1%}")
    print(f"  Break-even WR @ mid  : {be_at_mid:.1%}")
    print(f"  Observed WR          : {overall_wr:.1%}")
    margin_vs_ask = overall_wr - be_at_ask
    margin_vs_mid = overall_wr - be_at_mid
    print(f"  Edge vs ask entry    : {color_pct(margin_vs_ask * 100)}")
    print(f"  Edge vs mid entry    : {color_pct(margin_vs_mid * 100)}")

    print(bold(f"\n  Suggested ALT_ENTRY_DISCOUNT_FRACTION : {cyan(frac)}"))
    print(dim("\n  To use spread-normalized discounts, set in .env:"))
    print(dim("    ALT_ENTRY_USE_SPREAD_DISCOUNT=true"))
    print(dim("    ALT_ENTRY_DISCOUNT_FRACTION=<value from above>"))
    print(dim("  Or pass to the bot: --discount-mode spread --discount-fraction <value>"))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate limit-order discount settings from trade history",
    )
    parser.add_argument(
        "--history",
        default=os.getenv("HISTORY_FILE", "trade_history_full.json"),
        metavar="PATH",
        help="Path to trade_history_full.json (default: trade_history_full.json)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=3,
        metavar="N",
        help="Minimum trades per group to include in analysis (default: 3)",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "all"],
        default="all",
        help="Filter trades by mode (default: all)",
    )
    args = parser.parse_args()

    if not Path(args.history).exists():
        print(red(f"Error: history file not found: {args.history}"))
        sys.exit(1)

    print(bold(f"\nLoading trade history from {args.history} ..."))
    records = load_history(args.history)

    if args.mode != "all":
        records = [r for r in records if r.mode == args.mode]

    if not records:
        print(red("No settled trades found. Run the bot first to accumulate history."))
        sys.exit(0)

    print(f"  Loaded {len(records)} settled trades (mode={args.mode})")

    # Run all analyses
    spread_groups = group_by_spread(records)
    ask_groups = group_by_ask(records)
    hour_groups = group_by_hour(records)

    print_spread_analysis(spread_groups, args.min_trades)
    print_ask_analysis(ask_groups, args.min_trades)
    print_hour_analysis(hour_groups, args.min_trades)
    print_mid_simulation(records)
    print_recommendations(records, args.min_trades)

    print()


if __name__ == "__main__":
    main()
