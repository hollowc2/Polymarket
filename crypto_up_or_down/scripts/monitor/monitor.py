#!/usr/bin/env python3
# /// script
# dependencies = ["rich>=13.0", "python-dotenv>=1.0", "requests>=2.28"]
# ///
"""Live dashboard for the Polymarket streak bot.

Usage:
    uv run python scripts/monitor.py          # one-shot (good with: watch -n 10)
    uv run python scripts/monitor.py --live   # auto-refresh every 10s (Ctrl-C to exit)
    uv run python scripts/monitor.py --live --interval 5
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.columns import Columns  # type: ignore[import-untyped]
from rich.console import Console, Group  # type: ignore[import-untyped]
from rich.live import Live  # type: ignore[import-untyped]
from rich.panel import Panel  # type: ignore[import-untyped]
from rich.table import Table  # type: ignore[import-untyped]
from rich.text import Text  # type: ignore[import-untyped]

# ── Config ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

ASSET = os.getenv("ASSET", "btc").upper()
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
PAPER_TRADE = os.getenv("PAPER_TRADE", "true").lower() == "true"
STRATEGY = os.getenv("STRATEGY", "streak_reversal")
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "50"))
MAX_DAILY_BETS = int(os.getenv("MAX_DAILY_BETS", "100"))
INITIAL_BANKROLL = float(os.getenv("INITIAL_BANKROLL", "100"))
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")
POLYGON_RPC_URL = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")


_USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
_NATIVE_USDC = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"


def fetch_wallet_balance() -> float | None:
    """Fetch live USDC balance (USDC.e + native) via JSON-RPC — no web3 needed."""
    if not WALLET_ADDRESS:
        return None
    addr_data = "000000000000000000000000" + WALLET_ADDRESS[2:].lower()

    def _call(token: str) -> int:
        resp = requests.post(
            POLYGON_RPC_URL,
            json={
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": token, "data": "0x70a08231" + addr_data}, "latest"],
                "id": 1,
            },
            timeout=5,
        )
        return int(resp.json().get("result", "0x0"), 16)

    try:
        return (_call(_USDC_E) + _call(_NATIVE_USDC)) / 1_000_000
    except Exception:
        return None


def _resolve(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, "")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else Path.cwd() / p
    return _PROJECT_ROOT / default


HISTORY_PATH = _resolve("HISTORY_FILE", "trade_history_full.json")
STATE_PATH = _resolve("TRADES_FILE", "trades.json")
LOG_PATH = _resolve("LOG_FILE", "bot.log")

# ── Data loading ──────────────────────────────────────────────────────────────


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "trades" in data:
        return data["trades"]
    return []


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    with STATE_PATH.open() as f:
        return json.load(f)


def tail_log(n: int = 4) -> list[str]:
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open() as f:
        lines = f.readlines()
    return [line.rstrip() for line in lines[-n:]]


# ── Stats ─────────────────────────────────────────────────────────────────────


def compute_stats(trades: list[dict]) -> dict:
    settled = [t for t in trades if t.get("settlement", {}).get("status") == "settled"]
    pending = [t for t in trades if t.get("settlement", {}).get("status") == "pending"]
    wins = [t for t in settled if t.get("settlement", {}).get("won")]
    losses = [t for t in settled if not t.get("settlement", {}).get("won")]

    win_pnls = [t["settlement"]["net_profit"] for t in wins]
    loss_pnls = [t["settlement"]["net_profit"] for t in losses]
    all_pnls = [t["settlement"]["net_profit"] for t in settled]

    return {
        "total": len(trades),
        "settled": len(settled),
        "pending": len(pending),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(settled) * 100 if settled else 0.0,
        "total_pnl": sum(all_pnls),
        "avg_win": sum(win_pnls) / len(win_pnls) if win_pnls else 0.0,
        "avg_loss": sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0,
        "largest_win": max(win_pnls) if win_pnls else 0.0,
        "largest_loss": min(loss_pnls) if loss_pnls else 0.0,
    }


# ── Dashboard builder ─────────────────────────────────────────────────────────


def build_dashboard() -> Group:
    trades = load_history()
    state = load_state()
    stats = compute_stats(trades)

    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    mode = "PAPER" if PAPER_TRADE else "LIVE"
    mode_style = "bold yellow" if PAPER_TRADE else "bold red"

    # Pull strategy name from last trade context if available
    last_strategy = STRATEGY
    if trades:
        last_strategy = trades[-1].get("context", {}).get("strategy", STRATEGY)

    # ── Header panel ──────────────────────────────────────────────────────────
    header_text = Text()
    header_text.append(f"Strategy: {last_strategy}  │  ", style="cyan")
    header_text.append(f"Asset: {ASSET}  │  ", style="cyan")
    header_text.append(f"Timeframe: {TIMEFRAME}  │  ", style="cyan")
    header_text.append(f"Mode: {mode}", style=mode_style)
    header_text.append(f"     {now_str}", style="dim")
    header_panel = Panel(header_text, title="[bold blue]Polymarket Bot Monitor[/bold blue]", border_style="blue")

    # ── Stats row ─────────────────────────────────────────────────────────────
    bankroll = state.get("bankroll", INITIAL_BANKROLL)
    daily_bets = state.get("daily_bets", 0)
    daily_pnl = float(state.get("daily_pnl", 0.0))
    last_reset = state.get("last_reset_date", "—")

    daily_loss_over = daily_pnl < 0 and abs(daily_pnl) >= MAX_DAILY_LOSS
    daily_bets_over = daily_bets >= MAX_DAILY_BETS
    trading_allowed = not daily_loss_over and not daily_bets_over

    # Bankroll panel — show live on-chain balance when running live
    br_grid = Table.grid(padding=(0, 0))
    if not PAPER_TRADE and WALLET_ADDRESS:
        live_bal = fetch_wallet_balance()
        if live_bal is not None:
            br_grid.add_row(Text(f"${live_bal:.2f}", style="bold white"))
            bankroll_panel = Panel(br_grid, title="[dim]WALLET (LIVE)[/dim]", border_style="green")
        else:
            br_grid.add_row(Text(f"${bankroll:.2f}", style="bold white"))
            br_grid.add_row(Text("chain read failed", style="dim red"))
            bankroll_panel = Panel(br_grid, title="[dim]WALLET (LIVE)[/dim]", border_style="dim")
    else:
        br_grid.add_row(Text(f"${bankroll:.2f}", style="bold white"))
        bankroll_panel = Panel(br_grid, title="[dim]BANKROLL[/dim]", border_style="dim")

    # Today panel
    today_grid = Table.grid(padding=(0, 1))
    today_grid.add_row(Text(f"Date:  {last_reset}", style="dim"))
    today_grid.add_row(Text(f"Bets:  {daily_bets}", style="bold"))
    pnl_style = "green" if daily_pnl >= 0 else "red"
    today_grid.add_row(Text(f"PnL:   {daily_pnl:+.2f}", style=pnl_style))
    today_panel = Panel(today_grid, title="[dim]TODAY[/dim]", border_style="dim")

    # Status panel
    status_grid = Table.grid(padding=(0, 0))
    if trading_allowed:
        status_grid.add_row(Text("✓ Trading allowed", style="green"))
    else:
        if daily_loss_over:
            status_grid.add_row(Text("✗ Daily loss limit hit", style="bold red"))
        if daily_bets_over:
            status_grid.add_row(Text("✗ Daily bet limit hit", style="bold red"))
    status_grid.add_row(Text(f"Loss:  ${abs(daily_pnl):.2f} / ${MAX_DAILY_LOSS:.2f}", style="dim"))
    status_grid.add_row(Text(f"Bets:  {daily_bets} / {MAX_DAILY_BETS}", style="dim"))
    status_panel = Panel(status_grid, title="[dim]STATUS[/dim]", border_style="dim")

    stats_row = Columns([bankroll_panel, today_panel, status_panel], equal=True, expand=True)

    # ── All-time performance panel ────────────────────────────────────────────
    pnl_color = "green" if stats["total_pnl"] >= 0 else "red"
    perf_text = Text()
    perf_text.append(f"Trades: {stats['total']}", style="white")
    perf_text.append("  │  ", style="dim")
    perf_text.append(f"Settled: {stats['settled']}", style="white")
    if stats["pending"] > 0:
        perf_text.append(f"  Pending: {stats['pending']}", style="yellow")
    perf_text.append("  │  ", style="dim")
    perf_text.append(f"Win Rate: {stats['win_rate']:.1f}%", style="cyan")
    perf_text.append("  │  ", style="dim")
    perf_text.append(f"PnL: {stats['total_pnl']:+.2f}", style=pnl_color)
    perf_text.append("  │  ", style="dim")
    perf_text.append(f"Avg W: {stats['avg_win']:+.2f}", style="green")
    perf_text.append("  ", style="dim")
    perf_text.append(f"Avg L: {stats['avg_loss']:+.2f}", style="red")
    if stats["largest_win"] or stats["largest_loss"]:
        perf_text.append("  │  ", style="dim")
        perf_text.append(f"Best: {stats['largest_win']:+.2f}", style="green")
        perf_text.append("  ", style="dim")
        perf_text.append(f"Worst: {stats['largest_loss']:+.2f}", style="red")
    perf_panel = Panel(perf_text, title="[dim]ALL-TIME PERFORMANCE[/dim]", border_style="dim")

    # ── Recent trades table ───────────────────────────────────────────────────
    recent = list(reversed(trades[-10:]))  # newest first

    trade_table = Table(
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        show_edge=False,
        pad_edge=False,
        expand=True,
    )
    trade_table.add_column("Time (UTC)", style="dim", min_width=17)
    trade_table.add_column("Direction", min_width=9)
    trade_table.add_column("Outcome", min_width=8)
    trade_table.add_column("PnL", justify="right", min_width=9)
    trade_table.add_column("Price", justify="right", min_width=6)
    trade_table.add_column("Status", min_width=9)

    for t in recent:
        pos = t.get("position", {})
        ex = t.get("execution", {})
        sett = t.get("settlement", {})

        exec_ts = ex.get("timestamp", 0)
        if exec_ts:
            dt = datetime.fromtimestamp(exec_ts / 1000, tz=UTC)
            time_str = dt.strftime("%m-%d %H:%M:%S")
        else:
            time_str = "—"

        direction = pos.get("direction", "?")
        if direction == "up":
            dir_text = Text("UP  ▲", style="green")
        elif direction == "down":
            dir_text = Text("DOWN ▼", style="red")
        else:
            dir_text = Text(direction or "—", style="dim")

        status = sett.get("status", "?")
        if status == "settled":
            won = sett.get("won")
            outcome = (sett.get("outcome") or "?").upper()
            if won is True:
                outcome_text = Text(f"{outcome} ✓", style="green")
            elif won is False:
                outcome_text = Text(f"{outcome} ✗", style="red")
            else:
                outcome_text = Text(outcome, style="dim")
        elif status == "pending":
            outcome_text = Text("pending", style="yellow")
        else:
            outcome_text = Text("—", style="dim")

        net_profit = sett.get("net_profit")
        if net_profit is not None:
            pnl_display = Text(f"${net_profit:+.2f}", style="green" if net_profit >= 0 else "red")
        else:
            pnl_display = Text("—", style="dim")

        fill_price = ex.get("fill_price") or 0.0
        price_str = f"{fill_price:.3f}" if fill_price else "—"

        trade_table.add_row(time_str, dir_text, outcome_text, pnl_display, price_str, status)

    trades_panel = Panel(
        trade_table,
        title=f"[dim]RECENT TRADES (last {min(10, len(trades))})[/dim]",
        border_style="dim",
    )

    # ── Log tail ──────────────────────────────────────────────────────────────
    log_lines = tail_log(4)
    renderables: list = [header_panel, stats_row, perf_panel, trades_panel]
    if log_lines:
        log_text = Text("\n".join(log_lines), style="dim", overflow="fold")
        renderables.append(Panel(log_text, title="[dim]BOT LOG (last lines)[/dim]", border_style="dim"))

    return Group(*renderables)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket bot monitor dashboard")
    parser.add_argument("--live", action="store_true", help="Auto-refresh every N seconds (Ctrl-C to exit)")
    parser.add_argument(
        "--interval", type=int, default=10, metavar="SEC", help="Refresh interval in seconds (default: 10)"
    )
    args = parser.parse_args()

    if args.live:
        with Live(build_dashboard(), refresh_per_second=0.1, screen=True) as live:
            try:
                while True:
                    time.sleep(args.interval)
                    live.update(build_dashboard())
            except KeyboardInterrupt:
                pass
    else:
        Console().print(build_dashboard())


if __name__ == "__main__":
    main()
