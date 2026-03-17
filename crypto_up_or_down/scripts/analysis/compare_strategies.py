"""Print a leaderboard table from backtest_results/summary.json.

Usage:
    uv run python scripts/compare_strategies.py
    uv run python scripts/compare_strategies.py --sort pnl
    uv run python scripts/compare_strategies.py --sort sharpe
    uv run python scripts/compare_strategies.py --sort trades

Sort options: win_rate (default), pnl, sharpe, trades
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ANSI colours
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

SORT_KEYS = {
    "win_rate": "win_rate",
    "pnl": "total_pnl",
    "sharpe": "sharpe_ratio",
    "trades": "trade_count",
}

SUMMARY_PATH = Path(__file__).resolve().parents[1] / "backtest_results" / "summary.json"


def colour_win_rate(win_rate: float) -> str:
    pct = f"{win_rate:.1%}"
    if win_rate >= 0.52:
        return f"{GREEN}{pct}{RESET}"
    if win_rate <= 0.48:
        return f"{RED}{pct}{RESET}"
    return pct


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backtest results across strategies")
    parser.add_argument(
        "--sort",
        default="win_rate",
        choices=list(SORT_KEYS.keys()),
        help="Sort column (default: win_rate)",
    )
    args = parser.parse_args()

    if not SUMMARY_PATH.exists():
        print(f"No summary found at {SUMMARY_PATH}. Run run_all_backtests.py first.")
        sys.exit(1)

    rows = json.loads(SUMMARY_PATH.read_text())
    if not rows:
        print("summary.json is empty.")
        sys.exit(0)

    sort_key = SORT_KEYS[args.sort]
    rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)

    # Column widths
    col_strategy = 22
    col_asset = 6
    col_tf = 5
    col_wr = 10
    col_pnl = 10
    col_dd = 10
    col_sharpe = 8
    col_trades = 7

    header = (
        f"{'Strategy':<{col_strategy}} "
        f"{'Asset':<{col_asset}} "
        f"{'TF':<{col_tf}} "
        f"{'Win Rate':>{col_wr}} "
        f"{'PnL':>{col_pnl}} "
        f"{'Drawdown':>{col_dd}} "
        f"{'Sharpe':>{col_sharpe}} "
        f"{'Trades':>{col_trades}}"
    )
    separator = "\u2500" * len(header)

    print(f"\n{header}")
    print(separator)

    for row in rows:
        strategy = row.get("strategy", "?")[:col_strategy]
        asset = row.get("asset", "?")
        tf = row.get("timeframe", "?")
        win_rate = float(row.get("win_rate", 0))
        pnl = float(row.get("total_pnl", 0))
        drawdown = float(row.get("max_drawdown", 0))
        sharpe = float(row.get("sharpe_ratio", 0))
        trade_count = int(row.get("trade_count", 0))

        wr_str = colour_win_rate(win_rate)
        # Coloured string has invisible escape chars — pad the visible portion
        visible_wr = f"{win_rate:.1%}"
        padding = col_wr - len(visible_wr)

        line = (
            f"{strategy:<{col_strategy}} "
            f"{asset:<{col_asset}} "
            f"{tf:<{col_tf}} "
            f"{' ' * padding}{wr_str} "
            f"{pnl:>+{col_pnl}.2f} "
            f"{drawdown:>{col_dd}.2f} "
            f"{sharpe:>{col_sharpe}.2f} "
            f"{trade_count:>{col_trades}}"
        )
        print(line)

    print()
    print(f"Sorted by: {args.sort}  |  {len(rows)} result(s)  |  {SUMMARY_PATH}")
    print()


if __name__ == "__main__":
    main()
