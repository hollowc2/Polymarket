#!/usr/bin/env python3
# /// script
# dependencies = ["rich>=13.0"]
# ///
"""All-6-bot leaderboard for live Polymarket bots.

Reads trade history files from the state directory and prints a Rich table
with risk-adjusted metrics for each bot — useful for go-live decisions.

Usage:
    uv run --script scripts/live_stats.py
    uv run --script scripts/live_stats.py --state-dir /opt/polymarket/state
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console  # type: ignore[import-untyped]
from rich.panel import Panel  # type: ignore[import-untyped]
from rich.table import Table  # type: ignore[import-untyped]
from rich.text import Text  # type: ignore[import-untyped]

# ── Bot registry ──────────────────────────────────────────────────────────────

BOTS = [
    {"label": "streak-bot (baseline)", "file": "trade_history_full.json"},
    {"label": "adx-eth-5m", "file": "adx-eth-5m-history.json"},
    {"label": "hl-momo-btc-5m", "file": "hl-orderflow-momo-5m-history.json"},
    {"label": "hl-momo-btc-15m", "file": "hl-orderflow-momo-15m-history.json"},
    {"label": "hl-reversal-btc-5m", "file": "hl-orderflow-reversal-5m-history.json"},
    {"label": "3barmomo-hl-5m", "file": "3barmomo-hl-5m-history.json"},
    {"label": "pinbar-hl-5m", "file": "pinbar-hl-5m-history.json"},
    {"label": "delta-flip-btc-5m", "file": "delta-flip-btc-5m-history.json"},
]

# TurtleQuant / SlowQuant use open/close event format, not settled trades
TURTLEQUANT_POSITIONS_FILE = "turtlequant/turtlequant-positions.json"
TURTLEQUANT_HISTORY_FILE = "turtlequant/turtlequant-history.json"
SLOWQUANT_POSITIONS_FILE = "slowquant/slowquant-positions.json"
SLOWQUANT_HISTORY_FILE = "slowquant/slowquant-history.json"

DEFAULT_STATE_DIR = "/opt/polymarket/state"

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    fill_price: float
    won: bool
    net_profit: float
    timestamp: int  # ms epoch, for consecutive-loss ordering


def load_history(path: Path) -> list[TradeRecord]:
    if not path.exists():
        return []
    with path.open() as f:
        raw: list[dict] = json.load(f)

    records: list[TradeRecord] = []
    for item in raw:
        try:
            exec_ = item.get("execution", {})
            settlement = item.get("settlement", {})

            if settlement.get("status") not in ("settled", "forced_exit"):
                continue
            if settlement.get("won") is None:
                continue

            records.append(
                TradeRecord(
                    fill_price=float(exec_.get("fill_price", exec_.get("entry_price", 0.5))),
                    won=bool(settlement.get("won", False)),
                    net_profit=float(settlement.get("net_profit", 0.0)),
                    timestamp=int(exec_.get("timestamp", 0)),
                )
            )
        except Exception:
            continue

    return records


def load_raw_history(path: Path) -> list[dict]:
    """Load raw trade JSON (preserves gate sub-dict and all fields)."""
    if not path.exists():
        return []
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return []


# ── Stats ─────────────────────────────────────────────────────────────────────


def _wilson_margin(wins: int, n: int) -> float:
    """95% Wilson score CI half-width."""
    if n == 0:
        return 0.0
    z = 1.96
    margin = z * math.sqrt(wins * (n - wins) / n + z**2 / 4) / (n + z**2)
    return margin


POLYMARKET_FEE_RATE_BPS = 200  # Polymarket standard taker fee (2%)


def _breakeven_wr(avg_fill: float) -> float:
    """Break-even win rate given average fill price and Polymarket fee.

    Formula: BE = fill / ((1 - fill) * (1 - fee_pct) + fill)
    where fee_pct = (rate_bps/10000) * fill * (1-fill)
    """
    if avg_fill <= 0:
        return 1.0
    fee_pct = (POLYMARKET_FEE_RATE_BPS / 10000) * avg_fill * (1 - avg_fill)
    return avg_fill / ((1 - avg_fill) * (1 - fee_pct) + avg_fill)


@dataclass
class BotStats:
    label: str
    trades: list[TradeRecord] = field(default_factory=list)
    missing: bool = False  # history file not found

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.won)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def avg_fill(self) -> float:
        valid = [t.fill_price for t in self.trades if t.fill_price > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def be_wr(self) -> float:
        return _breakeven_wr(self.avg_fill)

    @property
    def edge(self) -> float:
        return self.win_rate - self.be_wr

    @property
    def pnls(self) -> list[float]:
        return [t.net_profit for t in self.trades]

    @property
    def total_pnl(self) -> float:
        return sum(self.pnls)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.n if self.n else 0.0

    @property
    def sharpe(self) -> float:
        if self.n < 2:
            return 0.0
        ps = self.pnls
        mean = sum(ps) / self.n
        variance = sum((p - mean) ** 2 for p in ps) / (self.n - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        return mean / std * math.sqrt(self.n)

    @property
    def max_drawdown(self) -> float:
        peak = 0.0
        max_dd = 0.0
        cumulative = 0.0
        for pnl in self.pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            max_dd = min(max_dd, cumulative - peak)
        return max_dd

    @property
    def max_consec_losses(self) -> int:
        sorted_trades = sorted(self.trades, key=lambda t: t.timestamp)
        max_cl = cur = 0
        for t in sorted_trades:
            if not t.won:
                cur += 1
                max_cl = max(max_cl, cur)
            else:
                cur = 0
        return max_cl


# ── Rich formatting helpers ───────────────────────────────────────────────────

_DASH = Text("—", style="dim")


def _n_cell(n: int) -> Text:
    if n == 0:
        return Text("0", style="dim")
    if n < 30:
        return Text(f"⚠ {n}", style="yellow")
    return Text(str(n))


def _pct(val: float) -> Text:
    return Text(f"{val:.1%}")


def _ci(margin: float) -> Text:
    return Text(f"±{margin:.1%}", style="dim")


def _edge(val: float) -> Text:
    s = f"{val:+.1%}"
    if val > 0:
        return Text(s, style="bold green")
    if val < 0:
        return Text(s, style="bold red")
    return Text(s, style="yellow")


def _money(val: float) -> Text:
    s = f"${val:+.2f}"
    style = "green" if val > 0 else ("red" if val < 0 else "dim")
    return Text(s, style=style)


def _sharpe(val: float) -> Text:
    s = f"{val:.2f}"
    if val >= 1.0:
        return Text(s, style="bold green")
    if val >= 0:
        return Text(s, style="yellow")
    return Text(s, style="red")


def _drawdown(val: float) -> Text:
    if val == 0.0:
        return Text("$0.00", style="dim")
    return Text(f"${val:.2f}", style="red")


# ── Table builder ─────────────────────────────────────────────────────────────


def build_table(stats: list[BotStats]) -> Table:
    table = Table(
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        show_edge=True,
        pad_edge=True,
        expand=True,
    )
    table.add_column("Bot", min_width=16, style="bold cyan")
    table.add_column("n", justify="right", min_width=6)
    table.add_column("Win%", justify="right", min_width=6)
    table.add_column("± CI", justify="right", min_width=6)
    table.add_column("BE%", justify="right", min_width=6)
    table.add_column("Edge", justify="right", min_width=7)
    table.add_column("PnL $", justify="right", min_width=9)
    table.add_column("Avg $/trade", justify="right", min_width=11)
    table.add_column("Sharpe", justify="right", min_width=7)
    table.add_column("MaxDD $", justify="right", min_width=9)
    table.add_column("MaxCL", justify="right", min_width=6)

    for s in stats:
        if s.missing:
            table.add_row(
                s.label,
                Text("(no file)", style="dim"),
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
            )
            continue

        if s.n == 0:
            table.add_row(
                s.label,
                _n_cell(0),
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
                _DASH,
            )
            continue

        margin = _wilson_margin(s.wins, s.n)
        table.add_row(
            s.label,
            _n_cell(s.n),
            _pct(s.win_rate),
            _ci(margin),
            _pct(s.be_wr),
            _edge(s.edge),
            _money(s.total_pnl),
            _money(s.avg_pnl),
            _sharpe(s.sharpe),
            _drawdown(s.max_drawdown),
            Text(str(s.max_consec_losses)),
        )

    return table


# ── TurtleQuant positions panel ───────────────────────────────────────────────


def build_turtlequant_panel(state_dir: Path) -> Table | None:
    """Build a Rich table showing open TurtleQuant positions and recent closes."""
    pos_path = state_dir / TURTLEQUANT_POSITIONS_FILE
    hist_path = state_dir / TURTLEQUANT_HISTORY_FILE

    if not pos_path.exists() and not hist_path.exists():
        return None

    # Load open positions
    positions: list[dict] = []
    nav = 1000.0
    if pos_path.exists():
        try:
            data = json.loads(pos_path.read_text())
            nav = data.get("nav", 1000.0)
            positions = data.get("positions", [])
        except Exception:
            pass

    # Load recent close events from history
    closes: list[dict] = []
    if hist_path.exists():
        try:
            history = json.loads(hist_path.read_text())
            closes = [e for e in history if e.get("event") == "close"][-5:]
        except Exception:
            pass

    table = Table(
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        show_edge=True,
        pad_edge=True,
        expand=True,
        title="[bold cyan]TurtleQuant — Open Positions[/bold cyan]",
    )
    table.add_column("Question", min_width=40)
    table.add_column("Type", min_width=12)
    table.add_column("Strike", justify="right", min_width=9)
    table.add_column("Expiry", min_width=11)
    table.add_column("Entry", justify="right", min_width=7)
    table.add_column("Model P", justify="right", min_width=8)
    table.add_column("Edge@Entry", justify="right", min_width=10)
    table.add_column("Size $", justify="right", min_width=7)

    if not positions:
        table.add_row(Text("(no open positions)", style="dim"), "", "", "", "", "", "", "")
    else:
        for p in positions:
            edge = p.get("edge_at_entry", 0.0)
            edge_text = Text(f"{edge:+.3f}", style="bold green" if edge > 0.05 else "yellow")
            exp = p.get("expiry_iso", "")[:10]
            strike = p.get("strike", 0)
            table.add_row(
                Text(p.get("question", "")[:55], style="white"),
                Text(p.get("option_type", ""), style="cyan"),
                Text(f"${strike:,.0f}"),
                Text(exp, style="dim"),
                Text(f"{p.get('entry_price', 0):.3f}"),
                Text(f"{p.get('model_prob_at_entry', 0):.3f}"),
                edge_text,
                Text(f"${p.get('size_usd', 0):.2f}"),
            )

    # NAV footer row
    table.add_section()
    table.add_row(
        Text(f"NAV: ${nav:.2f}  │  Open: {len(positions)}  │  Recent closes: {len(closes)}", style="dim"),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    )

    return table


def build_slowquant_panel(state_dir: Path) -> Table | None:
    """Build a Rich table showing open SlowQuant positions."""
    pos_path = state_dir / SLOWQUANT_POSITIONS_FILE
    hist_path = state_dir / SLOWQUANT_HISTORY_FILE

    if not pos_path.exists() and not hist_path.exists():
        return None

    positions: list[dict] = []
    nav = 1000.0
    if pos_path.exists():
        try:
            data = json.loads(pos_path.read_text())
            nav = data.get("nav", 1000.0)
            positions = data.get("positions", [])
        except Exception:
            pass

    closes: list[dict] = []
    if hist_path.exists():
        try:
            history = json.loads(hist_path.read_text())
            closes = [e for e in history if e.get("event") == "close"][-5:]
        except Exception:
            pass

    table = Table(
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        show_edge=True,
        pad_edge=True,
        expand=True,
        title="[bold magenta]SlowQuant — Open Positions[/bold magenta]",
    )
    table.add_column("Question", min_width=40)
    table.add_column("Type", min_width=12)
    table.add_column("Strike", justify="right", min_width=9)
    table.add_column("Expiry", min_width=11)
    table.add_column("Entry", justify="right", min_width=7)
    table.add_column("MC Prob", justify="right", min_width=8)
    table.add_column("Edge@Entry", justify="right", min_width=10)
    table.add_column("Days Left", justify="right", min_width=9)

    if not positions:
        table.add_row(Text("(no open positions)", style="dim"), "", "", "", "", "", "", "")
    else:
        for p in positions:
            edge = p.get("edge_at_entry", 0.0)
            edge_style = "bold green" if edge > 0.04 else "yellow"
            model_p = p.get("model_prob_at_entry", 0.0)
            exp = p.get("expiry_iso", "")[:10]
            # Days left
            days_left = "—"
            try:
                from datetime import UTC, datetime

                exp_dt = datetime.fromisoformat(p.get("expiry_iso", ""))
                dl = (exp_dt - datetime.now(UTC)).days
                days_left = f"{dl}d"
            except Exception:
                pass
            table.add_row(
                Text(p.get("question", "")[:55], style="white"),
                Text(p.get("option_type", ""), style="cyan"),
                Text(f"${p.get('strike', 0):,.0f}"),
                Text(exp, style="dim"),
                Text(f"{p.get('entry_price', 0):.3f}"),
                Text(f"{model_p:.3f}"),
                Text(f"{edge:+.3f}", style=edge_style),
                Text(days_left, style="dim"),
            )

    table.add_section()
    table.add_row(
        Text(f"NAV: ${nav:.2f}  │  Open: {len(positions)}  │  Recent closes: {len(closes)}", style="dim"),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    )
    return table


# ── Gate boost analysis panel ─────────────────────────────────────────────────


def build_gate_boost_panel(label: str, raw: list[dict]) -> Table | None:
    """Build a gate boost analysis table for a bot that has vol_accel gate data.

    Only shown when at least one settled trade has gate.name containing 'vol_accel'.
    Segments trades into: boosted, not-boosted, and skipped.
    """
    # Filter settled trades that have gate metadata
    gate_trades = [
        t
        for t in raw
        if t.get("gate")
        and "vol_accel" in t["gate"].get("name", "")
        and t.get("settlement", {}).get("status") in ("settled", "forced_exit")
        and t.get("settlement", {}).get("won") is not None
    ]
    if not gate_trades:
        return None

    def _seg_stats(items: list[dict]) -> tuple[int, float | None, float | None]:
        """Return (count, win_rate, net_pnl) for a segment."""
        n = len(items)
        if n == 0:
            return 0, None, None
        wins = sum(1 for t in items if t.get("settlement", {}).get("won"))
        pnl = sum(t.get("settlement", {}).get("net_profit", 0.0) for t in items)
        return n, wins / n, pnl

    boosted = [t for t in gate_trades if t["gate"].get("boosted")]
    not_boosted = [t for t in gate_trades if not t["gate"].get("boosted") and not t["gate"].get("skipped")]
    skipped_all = [
        t for t in raw if t.get("gate") and "vol_accel" in t["gate"].get("name", "") and t["gate"].get("skipped")
    ]

    table = Table(
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        show_edge=True,
        pad_edge=True,
        expand=True,
        title=f"[bold yellow]VolAccelGate — Boost Analysis  ({label})[/bold yellow]",
    )
    table.add_column("Condition", min_width=18)
    table.add_column("Trades", justify="right", min_width=7)
    table.add_column("Win%", justify="right", min_width=7)
    table.add_column("Net P&L", justify="right", min_width=10)
    table.add_column("Skipped", justify="right", min_width=8)

    def _row(condition: str, items: list[dict], skipped_count: int = 0) -> None:
        n, wr, pnl = _seg_stats(items)
        wr_text = _pct(wr) if wr is not None else _DASH
        pnl_text = _money(pnl) if pnl is not None else _DASH
        skip_text = Text(str(skipped_count), style="yellow") if skipped_count else Text("0", style="dim")
        table.add_row(Text(condition), _n_cell(n), wr_text, pnl_text, skip_text)

    _row("gate_boosted=T", boosted)
    _row("gate_boosted=F", not_boosted)
    _row("gate_skipped=T", [], skipped_count=len(skipped_all))

    return table


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="All-bot leaderboard — go-live decision aid")
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        metavar="DIR",
        help=f"Directory containing bot state files (default: {DEFAULT_STATE_DIR})",
    )
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    all_stats: list[BotStats] = []
    raw_by_bot: list[tuple[str, list[dict]]] = []
    for bot in BOTS:
        path = state_dir / bot["file"]
        if not path.exists():
            all_stats.append(BotStats(label=bot["label"], missing=True))
            raw_by_bot.append((bot["label"], []))
        else:
            raw = load_raw_history(path)
            all_stats.append(BotStats(label=bot["label"], trades=load_history(path)))
            raw_by_bot.append((bot["label"], raw))

    total_trades = sum(s.n for s in all_stats)
    files_found = sum(1 for s in all_stats if not s.missing)

    console = Console()

    summary = Text()
    summary.append(f"State dir: {state_dir}  │  ", style="dim")
    summary.append(f"Files found: {files_found}/{len(BOTS)}  │  ", style="dim")
    summary.append(f"Total settled trades: {total_trades}  │  ")
    summary.append(now_str, style="dim")

    table_panel = Panel(
        build_table(all_stats),
        title="[bold blue]Polymarket Bot Leaderboard[/bold blue]",
        border_style="dim",
    )

    footer = Text(
        "n < 30 = ⚠  results unreliable  │  CI = 95% Wilson  │  Sharpe = per-trade",
        style="dim",
        justify="center",
    )

    console.print()
    console.print(summary)
    console.print(table_panel)
    console.print(footer)

    # Gate boost analysis — only shown when a bot has vol_accel gate history
    for label, raw in raw_by_bot:
        gate_panel = build_gate_boost_panel(label, raw)
        if gate_panel is not None:
            console.print()
            console.print(gate_panel)

    tq_panel = build_turtlequant_panel(state_dir)
    if tq_panel is not None:
        console.print()
        console.print(tq_panel)

    sq_panel = build_slowquant_panel(state_dir)
    if sq_panel is not None:
        console.print()
        console.print(sq_panel)

    console.print()


if __name__ == "__main__":
    main()
