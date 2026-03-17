#!/usr/bin/env python3
# /// script
# dependencies = ["rich>=13.0"]
# ///
"""Compare alt-entry-bot (paper limit orders) vs streak-live-bot (live CLOB limit orders).

Both bots run the same strategy (streak_reversal/eth/5m, spread discount 0.5) but in
different modes. This script matches trades by market slug + direction to compare:
  - Signal overlap (how many markets both bots traded)
  - Fill price delta (paper simulation vs live CLOB fill)
  - Missed rate (orders that expired unfilled)
  - Win rate on shared signals
  - PnL comparison

Usage:
    # After scp'ing state files from VPS:
    scp srv1355486:/opt/polymarket/state/alt-entry-history.json /tmp/
    scp srv1355486:/opt/polymarket/state/streak-live-history.json /tmp/
    scp srv1355486:/opt/polymarket/state/alt-entry-missed.json /tmp/      # optional
    scp srv1355486:/opt/polymarket/state/streak-live-missed.json /tmp/    # optional

    uv run --script scripts/compare_entries.py \\
        --alt /tmp/alt-entry-history.json \\
        --live /tmp/streak-live-history.json

    # With missed-orders files (recommended — enables missed detail section + accurate miss rates):
    uv run --script scripts/compare_entries.py \\
        --alt /tmp/alt-entry-history.json \\
        --live /tmp/streak-live-history.json \\
        --alt-missed /tmp/alt-entry-missed.json \\
        --live-missed /tmp/streak-live-missed.json

    # Alt history is automatically trimmed to the live bot's active window
    # so trade counts are comparable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console  # type: ignore[import-untyped]
from rich.panel import Panel  # type: ignore[import-untyped]
from rich.rule import Rule  # type: ignore[import-untyped]
from rich.table import Table  # type: ignore[import-untyped]
from rich.text import Text  # type: ignore[import-untyped]

console = Console()

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class TradeEntry:
    slug: str
    direction: str
    fill_price: float
    best_ask: float
    entry_price: float
    won: bool | None
    net_profit: float
    settled: bool
    exec_timestamp: int


def load_history(path: Path) -> list[TradeEntry]:
    if not path.exists():
        console.print(f"[yellow]Warning: file not found: {path}[/yellow]")
        return []
    with path.open() as f:
        raw: list[dict] = json.load(f)

    entries: list[TradeEntry] = []
    for item in raw:
        try:
            market = item.get("market", {})
            position = item.get("position", {})
            execution = item.get("execution", {})
            settlement = item.get("settlement", {})

            slug = market.get("slug", "")
            direction = position.get("direction", "")
            if not slug or not direction:
                continue

            fill_price = float(execution.get("fill_price") or execution.get("entry_price") or 0.0)
            best_ask = float(execution.get("best_ask") or execution.get("entry_price") or 0.0)
            entry_price = float(execution.get("entry_price") or 0.0)
            exec_ts = int(execution.get("timestamp") or 0)

            status = settlement.get("status", "pending")
            settled = status in ("settled", "force_exit", "forced_exit")
            won = settlement.get("won")
            net_profit = float(settlement.get("net_profit") or 0.0)

            entries.append(
                TradeEntry(
                    slug=slug,
                    direction=direction,
                    fill_price=fill_price,
                    best_ask=best_ask,
                    entry_price=entry_price,
                    won=won,
                    net_profit=net_profit,
                    settled=settled,
                    exec_timestamp=exec_ts,
                )
            )
        except Exception:
            continue

    return entries


@dataclass
class MissedOrder:
    timestamp_ms: int
    slug: str
    direction: str
    limit_price: float
    ask_at_placement: float
    discount_applied: float
    fill_window_sec: int


def load_missed_detail(path: Path | None) -> list[MissedOrder]:
    if path is None or not path.exists():
        return []
    try:
        with path.open() as f:
            raw = json.load(f)
        return [
            MissedOrder(
                timestamp_ms=int(m.get("timestamp", 0)),
                slug=m.get("slug", ""),
                direction=m.get("direction", ""),
                limit_price=float(m.get("limit_price", 0)),
                ask_at_placement=float(m.get("ask_at_placement", 0)),
                discount_applied=float(m.get("discount_applied", 0)),
                fill_window_sec=int(m.get("fill_window_sec", 0)),
            )
            for m in raw
            if m.get("slug")
        ]
    except Exception:
        return []


# ── Bot stats container ───────────────────────────────────────────────────────


@dataclass
class BotData:
    label: str
    trades: list[TradeEntry] = field(default_factory=list)
    missed: list[MissedOrder] = field(default_factory=list)

    @property
    def missed_count(self) -> int:
        return len(self.missed)

    @property
    def settled(self) -> list[TradeEntry]:
        return [t for t in self.trades if t.settled and t.won is not None]

    @property
    def pending(self) -> list[TradeEntry]:
        return [t for t in self.trades if not t.settled]

    @property
    def n_settled(self) -> int:
        return len(self.settled)

    @property
    def n_pending(self) -> int:
        return len(self.pending)

    @property
    def avg_fill(self) -> float:
        fills = [t.fill_price for t in self.settled if t.fill_price > 0]
        return sum(fills) / len(fills) if fills else 0.0

    @property
    def avg_ask(self) -> float:
        asks = [t.best_ask for t in self.settled if t.best_ask > 0]
        return sum(asks) / len(asks) if asks else 0.0

    @property
    def avg_discount(self) -> float:
        """Average discount below ask: positive = cheaper fill."""
        pairs = [(t.best_ask - t.fill_price) for t in self.settled if t.best_ask > 0 and t.fill_price > 0]
        return sum(pairs) / len(pairs) if pairs else 0.0

    @property
    def win_rate(self) -> float:
        s = self.settled
        return sum(1 for t in s if t.won) / len(s) if s else 0.0

    @property
    def wins(self) -> int:
        return sum(1 for t in self.settled if t.won)

    @property
    def total_pnl(self) -> float:
        return sum(t.net_profit for t in self.settled)

    @property
    def miss_rate(self) -> float:
        denom = self.n_settled + self.missed_count
        return self.missed_count / denom if denom > 0 else 0.0


# ── Trade matching ────────────────────────────────────────────────────────────


@dataclass
class MatchedPair:
    slug: str
    direction: str
    alt: TradeEntry
    live: TradeEntry


def match_trades(alt_trades: list[TradeEntry], live_trades: list[TradeEntry]) -> list[MatchedPair]:
    """Match by (slug, direction) — exact overlap. Takes first occurrence per key."""
    live_index: dict[tuple[str, str], TradeEntry] = {}
    for t in live_trades:
        key = (t.slug, t.direction)
        if key not in live_index:
            live_index[key] = t

    pairs: list[MatchedPair] = []
    seen: set[tuple[str, str]] = set()
    for t in alt_trades:
        key = (t.slug, t.direction)
        if key in live_index and key not in seen:
            pairs.append(MatchedPair(slug=t.slug, direction=t.direction, alt=t, live=live_index[key]))
            seen.add(key)

    return pairs


# ── Rich helpers ───────────────────────────────────────────────────────────────

_DASH = Text("—", style="dim")


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _price(v: float) -> Text:
    return Text(f"{v:.4f}")


def _delta(v: float, invert: bool = False) -> Text:
    """Render a price delta. Green = better for buyer (cheaper fill)."""
    s = f"{v:+.4f}"
    positive_is_good = not invert
    if v > 0:
        style = "green" if positive_is_good else "red"
    elif v < 0:
        style = "red" if positive_is_good else "green"
    else:
        style = "dim"
    return Text(s, style=style)


def _money(v: float) -> Text:
    s = f"${v:+.2f}"
    style = "green" if v > 0 else ("red" if v < 0 else "dim")
    return Text(s, style=style)


def _wr(wins: int, n: int) -> str:
    if n == 0:
        return "—"
    z = 1.96
    wr = wins / n
    margin = z * math.sqrt(wins * (n - wins) / n + z**2 / 4) / (n + z**2)
    return f"{wr:.1%} ±{margin:.1%}"


# ── Section printers ──────────────────────────────────────────────────────────


def print_overlap(alt: BotData, live: BotData, pairs: list[MatchedPair]) -> None:
    settled_pairs = [
        p for p in pairs if p.alt.settled and p.live.settled and p.alt.won is not None and p.live.won is not None
    ]

    alt_total = alt.n_settled + alt.missed_count
    live_total = live.n_settled + live.missed_count

    table = Table(show_header=True, header_style="bold dim", border_style="dim", expand=True)
    table.add_column("", style="bold cyan", min_width=22)
    table.add_column("alt-entry (paper)", justify="right", min_width=20)
    table.add_column("streak-live (real)", justify="right", min_width=20)

    table.add_row("Records in file", str(len(alt.trades)), str(len(live.trades)))
    table.add_row("Settled trades", str(alt.n_settled), str(live.n_settled))
    table.add_row("Pending trades", str(alt.n_pending), str(live.n_pending))
    table.add_row("Missed (expired)", str(alt.missed_count), str(live.missed_count))

    alt_overlap_pct = f"{len(pairs) / alt_total:.0%}" if alt_total > 0 else "—"
    live_overlap_pct = f"{len(pairs) / live_total:.0%}" if live_total > 0 else "—"
    table.add_row(
        "Matched markets (total)",
        f"{len(pairs)}  ({alt_overlap_pct} of alt)",
        f"{len(pairs)}  ({live_overlap_pct} of live)",
    )
    table.add_row("Matched + both settled", str(len(settled_pairs)), str(len(settled_pairs)))

    console.print(Panel(table, title="[bold]Signal Overlap[/bold]", border_style="blue"))


def print_fill_comparison(pairs: list[MatchedPair]) -> None:
    settled = [p for p in pairs if p.alt.settled and p.live.settled and p.alt.fill_price > 0 and p.live.fill_price > 0]

    if not settled:
        console.print(
            Panel(
                Text("No matched settled pairs.", style="dim"),
                title="[bold]Fill Price Comparison (matched markets)[/bold]",
                border_style="blue",
            )
        )
        return

    alt_fills = [p.alt.fill_price for p in settled]
    live_fills = [p.live.fill_price for p in settled]
    alt_asks = [p.alt.best_ask for p in settled if p.alt.best_ask > 0]
    live_asks = [p.live.best_ask for p in settled if p.live.best_ask > 0]

    alt_avg_fill = sum(alt_fills) / len(alt_fills)
    live_avg_fill = sum(live_fills) / len(live_fills)
    alt_avg_ask = sum(alt_asks) / len(alt_asks) if alt_asks else 0.0
    live_avg_ask = sum(live_asks) / len(live_asks) if live_asks else 0.0
    alt_avg_disc = alt_avg_ask - alt_avg_fill
    live_avg_disc = live_avg_ask - live_avg_fill

    # Per-pair: positive = live paid more than paper (bad for live)
    pair_deltas = [p.live.fill_price - p.alt.fill_price for p in settled]
    avg_pair_delta = sum(pair_deltas) / len(pair_deltas)
    live_cheaper = sum(1 for d in pair_deltas if d < 0)
    alt_cheaper = sum(1 for d in pair_deltas if d > 0)
    equal = sum(1 for d in pair_deltas if d == 0)

    table = Table(show_header=True, header_style="bold dim", border_style="dim", expand=True)
    table.add_column("Metric", style="bold cyan", min_width=26)
    table.add_column("alt-entry (paper)", justify="right", min_width=20)
    table.add_column("streak-live (real)", justify="right", min_width=20)
    table.add_column("delta (live − alt)", justify="right", min_width=18)

    table.add_row(
        "Avg fill price",
        _price(alt_avg_fill),
        _price(live_avg_fill),
        _delta(live_avg_fill - alt_avg_fill, invert=True),  # lower = better for buyer
    )
    table.add_row(
        "Avg ask at signal",
        _price(alt_avg_ask),
        _price(live_avg_ask),
        _delta(live_avg_ask - alt_avg_ask, invert=True),
    )
    table.add_row(
        "Avg discount vs ask",
        _delta(alt_avg_disc),
        _delta(live_avg_disc),
        _DASH,
    )
    table.add_row(
        "Avg per-pair delta",
        _DASH,
        _DASH,
        _delta(avg_pair_delta, invert=True),
    )
    table.add_row(
        "n pairs compared",
        Text(str(len(settled))),
        Text(str(len(settled))),
        _DASH,
    )

    console.print(Panel(table, title="[bold]Fill Price Comparison (matched markets)[/bold]", border_style="blue"))

    console.print(
        f"  Per-pair breakdown: live cheaper={live_cheaper}  alt cheaper={alt_cheaper}  equal={equal}  "
        f"(of {len(settled)} matched settled pairs)"
    )
    console.print()


def print_missed_rate(alt: BotData, live: BotData) -> None:
    table = Table(show_header=True, header_style="bold dim", border_style="dim", expand=True)
    table.add_column("", style="bold cyan", min_width=22)
    table.add_column("alt-entry (paper)", justify="right", min_width=20)
    table.add_column("streak-live (real)", justify="right", min_width=20)

    alt_denom = alt.n_settled + alt.missed_count
    live_denom = live.n_settled + live.missed_count

    table.add_row("Missed (expired orders)", str(alt.missed_count), str(live.missed_count))
    table.add_row("Settled", str(alt.n_settled), str(live.n_settled))
    table.add_row(
        "Miss rate (missed / settled+missed)",
        f"{alt.miss_rate:.1%}" if alt_denom > 0 else "—",
        f"{live.miss_rate:.1%}" if live_denom > 0 else "—",
    )

    note = ""
    if alt.missed_count == 0 and live.missed_count == 0:
        note = "\n  [dim]Pass --alt-missed / --live-missed for accurate miss counts.[/dim]"

    console.print(Panel(table, title="[bold]Missed Orders[/bold]", border_style="blue"))
    if note:
        console.print(Text.from_markup(note))


def print_missed_detail(live_missed: list[MissedOrder], alt_trades: list[TradeEntry]) -> None:
    """Per-missed-order breakdown: limit price, discount, and what alt-entry got on the same market."""
    if not live_missed:
        return

    from datetime import UTC, datetime

    alt_by_slug: dict[str, TradeEntry] = {}
    for t in alt_trades:
        if t.slug not in alt_by_slug:
            alt_by_slug[t.slug] = t

    table = Table(show_header=True, header_style="bold dim", border_style="dim", expand=True)
    table.add_column("Date (UTC)", style="dim", min_width=12)
    table.add_column("Dir", min_width=4)
    table.add_column("Ask", justify="right", min_width=6)
    table.add_column("Limit", justify="right", min_width=6)
    table.add_column("Disc%", justify="right", min_width=6)
    table.add_column("Window", justify="right", min_width=7)
    table.add_column("Alt outcome", justify="right", min_width=12)
    table.add_column("Est PnL ($2)", justify="right", min_width=12)

    total_impact = 0.0
    wins = losses = unknowns = 0

    for m in sorted(live_missed, key=lambda x: x.timestamp_ms):
        ts = datetime.fromtimestamp(m.timestamp_ms / 1000, tz=UTC)
        disc_pct = (m.ask_at_placement - m.limit_price) / m.ask_at_placement * 100 if m.ask_at_placement > 0 else 0

        alt_t = alt_by_slug.get(m.slug)
        if alt_t and alt_t.settled and alt_t.won is not None and alt_t.fill_price > 0:
            # Scale alt pnl to a $2 bet at the same fill price
            if alt_t.won:
                scaled_pnl = 2.0 * (1.0 / alt_t.fill_price - 1.0)
            else:
                scaled_pnl = -2.0
            total_impact += scaled_pnl
            if alt_t.won:
                wins += 1
            else:
                losses += 1
            outcome_label = Text("WIN", style="green") if alt_t.won else Text("LOSS", style="red")
            pnl_text = _money(scaled_pnl)
        else:
            outcome_label = Text("unknown", style="dim")
            pnl_text = Text("—", style="dim")
            unknowns += 1

        dir_text = Text(m.direction, style="cyan" if m.direction == "up" else "magenta")
        table.add_row(
            ts.strftime("%m-%d %H:%M"),
            dir_text,
            f"{m.ask_at_placement:.3f}",
            f"{m.limit_price:.3f}",
            f"{disc_pct:.1f}%",
            f"{m.fill_window_sec}s",
            outcome_label,
            pnl_text,
        )

    wr_str = f"   (win rate on missed: {wins / (wins + losses):.0%})" if (wins + losses) > 0 else ""
    pnl_color = "green" if total_impact >= 0 else "red"
    summary = (
        f"  Outcomes: [green]{wins} wins[/green] / [red]{losses} losses[/red] / [dim]{unknowns} unknown[/dim]"
        f"   Est. PnL impact: [{pnl_color}]${total_impact:+.2f}[/{pnl_color}]{wr_str}"
    )

    console.print(Panel(table, title="[bold]Missed Order Detail (live vs alt outcomes)[/bold]", border_style="yellow"))
    console.print(Text.from_markup(summary))
    console.print()


def print_win_rate(alt: BotData, live: BotData, pairs: list[MatchedPair]) -> None:
    settled_pairs = [
        p for p in pairs if p.alt.settled and p.live.settled and p.alt.won is not None and p.live.won is not None
    ]
    n_shared = len(settled_pairs)

    both_won = sum(1 for p in settled_pairs if p.alt.won and p.live.won)
    both_lost = sum(1 for p in settled_pairs if not p.alt.won and not p.live.won)
    alt_won_live_lost = sum(1 for p in settled_pairs if p.alt.won and not p.live.won)
    live_won_alt_lost = sum(1 for p in settled_pairs if not p.alt.won and p.live.won)

    alt_wins_shared = sum(1 for p in settled_pairs if p.alt.won)
    live_wins_shared = sum(1 for p in settled_pairs if p.live.won)

    table = Table(show_header=True, header_style="bold dim", border_style="dim", expand=True)
    table.add_column("Metric", style="bold cyan", min_width=26)
    table.add_column("alt-entry (paper)", justify="right", min_width=20)
    table.add_column("streak-live (real)", justify="right", min_width=20)

    table.add_row(
        "Win rate (shared mkts)",
        _wr(alt_wins_shared, n_shared),
        _wr(live_wins_shared, n_shared),
    )
    table.add_row(
        "Win rate (all settled)",
        _wr(alt.wins, alt.n_settled),
        _wr(live.wins, live.n_settled),
    )
    table.add_row("Shared settled markets", str(n_shared), str(n_shared))

    console.print(Panel(table, title="[bold]Win Rate[/bold]", border_style="blue"))

    if n_shared > 0:
        agreement = (both_won + both_lost) / n_shared
        console.print(Rule("Outcome concordance on shared markets", style="dim"))
        console.print(f"  Both won:           {both_won:>4}  ({both_won / n_shared:.1%})")
        console.print(f"  Both lost:          {both_lost:>4}  ({both_lost / n_shared:.1%})")
        console.print(f"  alt won, live lost: {alt_won_live_lost:>4}  ({alt_won_live_lost / n_shared:.1%})")
        console.print(f"  live won, alt lost: {live_won_alt_lost:>4}  ({live_won_alt_lost / n_shared:.1%})")
        console.print(f"  Agreement rate:     {agreement:.1%}  (signal quality check — should be high)")
        console.print()


def print_pnl(alt: BotData, live: BotData, pairs: list[MatchedPair]) -> None:
    settled_pairs = [p for p in pairs if p.alt.settled and p.live.settled and p.alt.won is not None]
    matched_keys = {(p.slug, p.direction) for p in settled_pairs}

    alt_matched_pnl = sum(p.alt.net_profit for p in settled_pairs)
    live_matched_pnl = sum(p.live.net_profit for p in settled_pairs)
    alt_unmatched_pnl = sum(t.net_profit for t in alt.settled if (t.slug, t.direction) not in matched_keys)
    live_unmatched_pnl = sum(t.net_profit for t in live.settled if (t.slug, t.direction) not in matched_keys)

    table = Table(show_header=True, header_style="bold dim", border_style="dim", expand=True)
    table.add_column("", style="bold cyan", min_width=26)
    table.add_column("alt-entry (paper)", justify="right", min_width=20)
    table.add_column("streak-live (real)", justify="right", min_width=20)

    table.add_row("Net PnL (all settled)", _money(alt.total_pnl), _money(live.total_pnl))
    table.add_row("Net PnL (matched mkts)", _money(alt_matched_pnl), _money(live_matched_pnl))
    table.add_row("Net PnL (unmatched)", _money(alt_unmatched_pnl), _money(live_unmatched_pnl))
    table.add_row(
        "Avg $/trade (settled)",
        _money(alt.total_pnl / alt.n_settled) if alt.n_settled else _DASH,
        _money(live.total_pnl / live.n_settled) if live.n_settled else _DASH,
    )
    table.add_row("n settled", str(alt.n_settled), str(live.n_settled))

    console.print(Panel(table, title="[bold]PnL Comparison[/bold]", border_style="blue"))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare alt-entry-bot (paper) vs streak-live-bot (live CLOB limit orders)"
    )
    parser.add_argument("--alt", required=True, metavar="FILE", help="alt-entry history JSON")
    parser.add_argument("--live", required=True, metavar="FILE", help="streak-live history JSON")
    parser.add_argument("--alt-missed", metavar="FILE", help="alt-entry missed_orders.json (optional)")
    parser.add_argument("--live-missed", metavar="FILE", help="streak-live missed_orders.json (optional)")
    args = parser.parse_args()

    alt_path = Path(args.alt)
    live_path = Path(args.live)
    alt_missed_path = Path(args.alt_missed) if args.alt_missed else None
    live_missed_path = Path(args.live_missed) if args.live_missed else None

    live_missed = load_missed_detail(live_missed_path)
    alt_missed = load_missed_detail(alt_missed_path)

    live_trades = load_history(live_path)
    all_alt_trades = load_history(alt_path)

    # Align alt to live bot's active window (first event = earliest of filled or missed)
    live_start_candidates = [t.exec_timestamp for t in live_trades if t.exec_timestamp > 0] + [
        m.timestamp_ms for m in live_missed
    ]
    live_start_ms = min(live_start_candidates) if live_start_candidates else 0
    # exec_timestamp for history records is in ms; missed.timestamp is also ms
    live_start_sec = live_start_ms / 1000 if live_start_ms > 1e10 else live_start_ms
    alt_trades = (
        [t for t in all_alt_trades if t.exec_timestamp / 1000 >= live_start_sec]
        if live_start_sec > 0
        else all_alt_trades
    )

    live = BotData(label="streak-live", trades=live_trades, missed=live_missed)
    alt = BotData(label="alt-entry", trades=alt_trades, missed=alt_missed)

    if not alt.trades and not live.trades:
        console.print("[red]Both history files are empty or not found.[/red]")
        return

    pairs = match_trades(alt.trades, live.trades)

    console.print()
    console.print(
        f"[bold blue]alt-entry  :[/bold blue] {alt_path}  "
        f"[dim]({len(all_alt_trades)} records total, {alt.n_settled} in live window)[/dim]"
    )
    console.print(
        f"[bold blue]streak-live:[/bold blue] {live_path}  "
        f"[dim]({len(live.trades)} records, {live.n_settled} settled)[/dim]"
    )
    if live_start_sec > 0:
        from datetime import UTC, datetime

        live_start_dt = datetime.fromtimestamp(live_start_sec, tz=UTC)
        console.print(f"[dim]Live window start: {live_start_dt.strftime('%Y-%m-%d %H:%M UTC')}[/dim]")
    console.print()

    print_overlap(alt, live, pairs)
    console.print()
    print_fill_comparison(pairs)
    print_missed_rate(alt, live)
    console.print()
    print_missed_detail(live_missed, alt.trades)
    print_win_rate(alt, live, pairs)
    print_pnl(alt, live, pairs)
    console.print()


if __name__ == "__main__":
    main()
