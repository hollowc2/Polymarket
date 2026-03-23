#!/usr/bin/env python3
# /// script
# dependencies = ["textual>=0.50.0", "rich>=13.0"]
# ///
"""Interactive TUI leaderboard for Polymarket bots.

Usage:
    uv run --script scripts/live_stats_tui.py
    uv run --script scripts/live_stats_tui.py --state-dir /opt/polymarket/state

Keys:
    Tab / Shift+Tab   Switch focus between panels
    r                 Force refresh
    q                 Quit
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rich.text import Text  # type: ignore[import-untyped]
from textual.app import App, ComposeResult  # type: ignore[import-untyped]
from textual.binding import Binding  # type: ignore[import-untyped]
from textual.containers import Container  # type: ignore[import-untyped]
from textual.widgets import DataTable, Footer, Static  # type: ignore[import-untyped]

# ── Bot discovery ─────────────────────────────────────────────────────────────

DEFAULT_STATE_DIR = "/opt/polymarket/state"

# Special-case filenames that don't follow the `{name}-history.json` convention
_SPECIAL_LABELS: dict[str, str] = {
    "trade_history_full.json": "streak-bot",
}

_container_cache: list[dict] = []
_container_cache_ts: float = 0.0
_CONTAINER_CACHE_TTL = 30.0  # seconds


def _label_from_filename(fname: str) -> str:
    if fname in _SPECIAL_LABELS:
        return _SPECIAL_LABELS[fname]
    return fname.removesuffix("-history.json")


def _bots_from_containers() -> list[dict]:
    """Query running polymarket-* containers for HISTORY_FILE env var (cached 30s)."""
    global _container_cache, _container_cache_ts
    now = time.monotonic()
    if now - _container_cache_ts < _CONTAINER_CACHE_TTL:
        return _container_cache
    result: list[dict] = []
    try:
        names = subprocess.check_output(
            ["docker", "ps", "--format", "{{.Names}}"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        for name in names:
            if not name.startswith("polymarket-"):
                continue
            env_out = subprocess.check_output(
                ["docker", "inspect", "--format",
                 "{{range .Config.Env}}{{println .}}{{end}}", name],
                timeout=3, stderr=subprocess.DEVNULL,
            ).decode()
            for line in env_out.splitlines():
                if line.startswith("HISTORY_FILE="):
                    fname = Path(line.split("=", 1)[1].strip()).name
                    result.append({"label": _label_from_filename(fname), "file": fname})
                    break
    except Exception:
        pass
    _container_cache = result
    _container_cache_ts = now
    return result


def discover_bots(state_dir: Path) -> list[dict]:
    """Return sorted bot list from running containers only.

    Containers are the source of truth for which bots are active.
    State files are used for data but don't create entries on their own.
    Container check is cached for 30 s so it doesn't block the TUI.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for b in _bots_from_containers():
        if b["label"] not in seen:
            seen.add(b["label"])
            unique.append(b)
    return sorted(unique, key=lambda b: b["label"])
REFRESH_SECONDS = 5

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    fill_price: float
    won: bool
    net_profit: float
    timestamp: int  # ms epoch
    market_slug: str = ""
    direction: str = ""
    amount: float = 0.0
    gate_name: str = ""
    gate_boosted: bool | None = None


def load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return []


def parse_trades(raw: list[dict]) -> list[TradeRecord]:
    records: list[TradeRecord] = []
    for item in raw:
        try:
            exec_ = item.get("execution", {})
            settlement = item.get("settlement", {})
            market = item.get("market", {})
            gate = item.get("gate") or {}
            position = item.get("position", {})

            if settlement.get("status") not in ("settled", "forced_exit"):
                continue
            if settlement.get("won") is None:
                continue

            ts = int(exec_.get("timestamp", 0) or market.get("timestamp", 0))
            records.append(
                TradeRecord(
                    fill_price=float(exec_.get("fill_price", exec_.get("entry_price", 0.5))),
                    won=bool(settlement.get("won", False)),
                    net_profit=float(settlement.get("net_profit", 0.0)),
                    timestamp=ts,
                    market_slug=str(market.get("slug", item.get("market", ""))),
                    direction=str(position.get("direction", "")),
                    amount=float(position.get("amount", 0.0)),
                    gate_name=gate.get("name", ""),
                    gate_boosted=gate.get("boosted") if gate else None,
                )
            )
        except Exception:
            continue
    return records


# ── Stats ─────────────────────────────────────────────────────────────────────

POLYMARKET_FEE_RATE_BPS = 200


def _wilson_margin(wins: int, n: int) -> float:
    if n == 0:
        return 0.0
    z = 1.96
    return z * math.sqrt(wins * (n - wins) / n + z**2 / 4) / (n + z**2)


def _breakeven_wr(avg_fill: float) -> float:
    if avg_fill <= 0:
        return 1.0
    fee_pct = (POLYMARKET_FEE_RATE_BPS / 10000) * avg_fill * (1 - avg_fill)
    return avg_fill / ((1 - avg_fill) * (1 - fee_pct) + avg_fill)


@dataclass
class BotStats:
    label: str
    trades: list[TradeRecord] = field(default_factory=list)
    missing: bool = False

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
        return 0.0 if std == 0 else mean / std * math.sqrt(self.n)

    @property
    def max_drawdown(self) -> float:
        peak = 0.0
        max_dd = 0.0
        cum = 0.0
        for pnl in self.pnls:
            cum += pnl
            peak = max(peak, cum)
            max_dd = min(max_dd, cum - peak)
        return max_dd

    @property
    def max_consec_losses(self) -> int:
        sorted_t = sorted(self.trades, key=lambda t: t.timestamp)
        mx = cur = 0
        for t in sorted_t:
            if not t.won:
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0
        return mx


# ── Rich cell helpers ─────────────────────────────────────────────────────────

_DASH = Text("—", style="dim")


def _t_n(n: int, missing: bool) -> Text:
    if missing:
        return Text("(no file)", style="dim")
    if n == 0:
        return Text("0", style="dim")
    if n < 30:
        return Text(f"⚠ {n}", style="bold yellow")
    return Text(str(n), style="white")


def _t_pct(val: float) -> Text:
    return Text(f"{val:.1%}", style="white")


def _t_ci(margin: float) -> Text:
    return Text(f"±{margin:.1%}", style="dim")


def _t_edge(val: float) -> Text:
    s = f"{val:+.1%}"
    if val > 0.02:
        return Text(s, style="bold green")
    if val > 0:
        return Text(s, style="green")
    if val < 0:
        return Text(s, style="bold red")
    return Text(s, style="yellow")


def _t_money(val: float) -> Text:
    s = f"${val:+.2f}"
    if val > 0:
        return Text(s, style="green")
    if val < 0:
        return Text(s, style="red")
    return Text(s, style="dim")


def _t_sharpe(val: float) -> Text:
    s = f"{val:.2f}"
    if val >= 1.0:
        return Text(s, style="bold green")
    if val >= 0:
        return Text(s, style="yellow")
    return Text(s, style="red")


def _t_dd(val: float) -> Text:
    if val == 0.0:
        return Text("$0.00", style="dim")
    return Text(f"${val:.2f}", style="red")


def _t_wl(won: bool) -> Text:
    return Text("  W  ", style="bold black on green") if won else Text("  L  ", style="bold white on red")


def _t_dir(direction: str) -> Text:
    if direction.upper() == "UP":
        return Text("UP", style="green")
    if direction.upper() == "DOWN":
        return Text("DOWN", style="red")
    return Text(direction or "—", style="dim")


def _t_cum(val: float) -> Text:
    s = f"${val:+.2f}"
    if val > 0:
        return Text(s, style="green")
    if val < 0:
        return Text(s, style="red")
    return Text(s, style="dim")


def _t_gate(gate_name: str, boosted: bool | None) -> Text:
    if not gate_name:
        return Text("—", style="dim")
    short = gate_name.replace("_gate", "").replace("_", " ")[:12]
    if boosted is True:
        return Text(f"{short} ↑", style="yellow")
    if boosted is False:
        return Text(f"{short} ·", style="dim")
    return Text(short, style="dim")


# ── TUI App ───────────────────────────────────────────────────────────────────

_APP_CSS = """
Screen {
    background: #0d0d0d;
}

#statusbar {
    height: 1;
    background: #111111;
    color: #555555;
    padding: 0 2;
}

#top-container {
    height: 45%;
    border: solid #2a2a2a;
    border-title-color: #ffd700;
    border-title-style: bold;
    border-title-align: left;
}

#bottom-container {
    height: 1fr;
    border: solid #2a2a2a;
    border-title-color: #ffd700;
    border-title-style: bold;
    border-title-align: left;
}

#leaderboard {
    background: #0d0d0d;
    color: #c0c0c0;
    height: 1fr;
    scrollbar-color: #333333;
    scrollbar-background: #0d0d0d;
}

#leaderboard > .datatable--header {
    background: #111111;
    color: #777777;
    text-style: bold;
}

#leaderboard:focus > .datatable--cursor {
    background: #0e2a46;
}

#leaderboard > .datatable--cursor {
    background: #0a1e30;
}

#trades {
    background: #0d0d0d;
    color: #c0c0c0;
    height: 1fr;
    scrollbar-color: #333333;
    scrollbar-background: #0d0d0d;
}

#trades > .datatable--header {
    background: #111111;
    color: #777777;
    text-style: bold;
}

#trades:focus > .datatable--cursor {
    background: #0e2a46;
}

#trades > .datatable--cursor {
    background: #0a1e30;
}

Footer {
    background: #111111;
    color: #555555;
}

Footer > .footer--key {
    background: #1a1a1a;
    color: #888888;
}
"""


class LiveStatsTUI(App):
    CSS = _APP_CSS

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "force_refresh", "Refresh"),
    ]

    def __init__(self, state_dir: Path) -> None:
        super().__init__()
        self.state_dir = state_dir
        self._all_stats: list[BotStats] = []
        self._selected_label: str = ""

    def compose(self) -> ComposeResult:
        yield Static("", id="statusbar")
        with Container(id="top-container"):
            yield DataTable(id="leaderboard", cursor_type="row", zebra_stripes=False)
        with Container(id="bottom-container"):
            yield DataTable(id="trades", cursor_type="row", zebra_stripes=False)
        yield Footer()

    def on_mount(self) -> None:
        self._setup_leaderboard_columns()
        self._setup_trades_columns()
        self.load_data()
        self.set_interval(REFRESH_SECONDS, self.load_data)

        # Set border titles
        self.query_one("#top-container", Container).border_title = " Polymarket Bot Leaderboard "
        self.query_one("#bottom-container", Container).border_title = f" {self._selected_label} — Trade History "

        # Focus leaderboard on start
        self.query_one("#leaderboard", DataTable).focus()

    def _setup_leaderboard_columns(self) -> None:
        t = self.query_one("#leaderboard", DataTable)
        t.add_columns(
            "Bot",
            "n",
            "Win%",
            "± CI",
            "BE%",
            "Edge",
            "PnL $",
            "Avg $/trade",
            "Sharpe",
            "MaxDD $",
            "MaxCL",
        )

    def _setup_trades_columns(self) -> None:
        t = self.query_one("#trades", DataTable)
        t.add_columns(
            "#",
            "Date/Time",
            "W/L",
            "Fill",
            "Net P&L",
            "Cum P&L",
            "Size $",
            "Dir",
            "Market Slug",
            "Gate",
        )

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self) -> None:
        bots = discover_bots(self.state_dir)
        stats: list[BotStats] = []
        for bot in bots:
            path = self.state_dir / bot["file"]
            raw = load_raw(path)
            trades = parse_trades(raw)
            stats.append(BotStats(label=bot["label"], trades=trades, missing=not path.exists()))
        self._all_stats = stats
        # Reset selection if the previously selected bot has disappeared
        if self._selected_label not in {s.label for s in stats} and stats:
            self._selected_label = stats[0].label
        self._redraw_leaderboard()
        self._redraw_trades()
        self._update_statusbar()

    def _redraw_leaderboard(self) -> None:
        table = self.query_one("#leaderboard", DataTable)
        prev_row = table.cursor_row
        table.clear()

        for s in self._all_stats:
            if s.missing or s.n == 0:
                table.add_row(
                    Text(s.label, style="dim cyan"),
                    _t_n(s.n, s.missing),
                    _DASH,
                    _DASH,
                    _DASH,
                    _DASH,
                    _DASH,
                    _DASH,
                    _DASH,
                    _DASH,
                    _DASH,
                    key=s.label,
                )
                continue

            margin = _wilson_margin(s.wins, s.n)
            table.add_row(
                Text(s.label, style="cyan"),
                _t_n(s.n, False),
                _t_pct(s.win_rate),
                _t_ci(margin),
                _t_pct(s.be_wr),
                _t_edge(s.edge),
                _t_money(s.total_pnl),
                _t_money(s.avg_pnl),
                _t_sharpe(s.sharpe),
                _t_dd(s.max_drawdown),
                Text(str(s.max_consec_losses), style="white" if s.max_consec_losses < 4 else "red"),
                key=s.label,
            )

        # Restore cursor; trigger detail sync
        row_count = len(self._all_stats)
        if row_count > 0:
            target = min(prev_row, row_count - 1)
            table.move_cursor(row=target)

    def _redraw_trades(self) -> None:
        stat = next((s for s in self._all_stats if s.label == self._selected_label), None)
        container = self.query_one("#bottom-container", Container)
        table = self.query_one("#trades", DataTable)

        if stat is None:
            container.border_title = f" {self._selected_label} — no data "
            table.clear()
            return

        # Build subtitle with key stats
        if stat.n > 0:
            warn = " ⚠ low n" if stat.n < 30 else ""
            subtitle = (
                f" {stat.label}  │  n={stat.n}{warn}  │  "
                f"Win {stat.win_rate:.1%}  │  "
                f"Edge {stat.edge:+.1%}  │  "
                f"PnL ${stat.total_pnl:+.2f}  │  "
                f"Sharpe {stat.sharpe:.2f} "
            )
        else:
            subtitle = f" {stat.label}  │  no settled trades "

        container.border_title = subtitle

        # Sort: most recent first
        sorted_desc = sorted(stat.trades, key=lambda t: t.timestamp, reverse=True)

        # Compute cumulative P&L in chronological order, then map to each trade
        sorted_asc = sorted(stat.trades, key=lambda t: t.timestamp)
        cum_by_id: dict[int, float] = {}
        running = 0.0
        for t in sorted_asc:
            running += t.net_profit
            cum_by_id[id(t)] = running

        table.clear()
        for i, t in enumerate(sorted_desc, 1):
            if t.timestamp:
                ts = datetime.fromtimestamp(t.timestamp / 1000, tz=UTC).strftime("%m-%d %H:%M")
            else:
                ts = "—"

            slug = t.market_slug
            # Show last 35 chars of slug for readability
            slug_display = ("…" + slug[-34:]) if len(slug) > 35 else slug

            table.add_row(
                Text(str(i), style="dim"),
                Text(ts, style="dim"),
                _t_wl(t.won),
                Text(f"{t.fill_price:.3f}", style="white"),
                _t_money(t.net_profit),
                _t_cum(cum_by_id.get(id(t), 0.0)),
                Text(f"${t.amount:.2f}" if t.amount else "—", style="white"),
                _t_dir(t.direction),
                Text(slug_display, style="dim"),
                _t_gate(t.gate_name, t.gate_boosted),
            )

    def _update_statusbar(self) -> None:
        bar = self.query_one("#statusbar", Static)
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        total = sum(s.n for s in self._all_stats)
        files = sum(1 for s in self._all_stats if not s.missing)
        bar.update(
            f"  State: {self.state_dir}  │  "
            f"Files: {files}/{len(self._all_stats)}  │  "
            f"Trades: {total}  │  "
            f"Refresh: {REFRESH_SECONDS}s  │  "
            f"{now}"
        )

    # ── Events ────────────────────────────────────────────────────────────────

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "leaderboard":
            return
        if event.row_key and event.row_key.value:
            label = str(event.row_key.value)
            if label != self._selected_label:
                self._selected_label = label
                self._redraw_trades()

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_force_refresh(self) -> None:
        self.load_data()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket bot TUI dashboard")
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        metavar="DIR",
        help=f"State directory (default: {DEFAULT_STATE_DIR})",
    )
    args = parser.parse_args()
    LiveStatsTUI(state_dir=Path(args.state_dir)).run()


if __name__ == "__main__":
    main()
