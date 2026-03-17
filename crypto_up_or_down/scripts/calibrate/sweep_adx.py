"""Sweep StreakADX strategy parameters with TrendFilter baked in.

Jointly sweeps adx_threshold × adx_period × trigger × ema_period (50, 100)
to find the parameter combination that maximises out-of-sample Sharpe.

The TrendFilter is fixed at mode=veto_with_trend (confirmed best mode from
prior gate sweep). ema_period is a first-class sweep dimension so both the
ema=50 and ema=100 results are directly comparable.

Usage:
    # Primary run — eth/5m, btc/5m, eth/15m
    uv run python scripts/sweep_adx.py

    # Single target
    uv run python scripts/sweep_adx.py --asset eth --tf 5m

    # With intrabar conservative scoring
    uv run python scripts/sweep_adx.py --resolution-mode intrabar_conservative

    # Disable CI sizing (match live bot behaviour)
    uv run python scripts/sweep_adx.py --no-ci-sizing

Walk-forward split: train < 2024-01-01, test >= 2024-01-01.
Scoring: close+spread with half_spread=0.005 (matches backtest convention).
Selection discipline: best config is chosen by TRAIN Sharpe; test Sharpe is
reported purely for validation — never used for selection.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from polymarket_algo.backtest.engine import _spike_candle_mask, score_resolution
from polymarket_algo.strategies.gates import TrendFilter
from polymarket_algo.strategies.streak_adx import StreakADXStrategy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "backtest_results"

CUTOFF = "2024-01-01"

# Sweep dimensions
TRIGGER_GRID = [3, 4, 5]
ADX_PERIOD_GRID = [7, 10, 14, 20]
ADX_THRESHOLD_GRID = [10, 15, 20, 25, 30]
EMA_PERIOD_GRID = [50, 100]  # both ema=50 (confirmed best for streak_reversal) and ema=100

# Default targets — primary + cross-asset
DEFAULT_TARGETS = [
    ("eth", "5m"),
    ("btc", "5m"),
    ("eth", "15m"),
]


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def load_candles(asset: str, tf: str) -> pd.DataFrame:
    path = DATA_DIR / f"{asset}_{tf}.parquet"
    df = pd.read_parquet(path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
    return df.sort_index()


def split_by_date(candles: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    train = candles.loc[candles.index < cutoff_ts].copy()
    test = candles.loc[candles.index >= cutoff_ts].copy()
    return train, test


# --------------------------------------------------------------------------- #
# Scoring (reused from sweep_gates.py pattern)
# --------------------------------------------------------------------------- #


def score_signals(
    signals: pd.DataFrame,
    candles: pd.DataFrame,
    resolution_mode: str = "close",
    half_spread: float = 0.005,
    atr_window: int = 20,
    spike_atr_mult: float = 2.5,
) -> dict[str, float]:
    sig = signals["signal"].astype(int)
    size = signals["size"].astype(float) if "size" in signals.columns else pd.Series(15.0, index=signals.index)

    next_close = candles["close"].shift(-1)
    next_high = candles["high"].shift(-1) if "high" in candles.columns else next_close
    next_low = candles["low"].shift(-1) if "low" in candles.columns else next_close

    mask = None
    if resolution_mode == "intrabar_conservative" and "high" in candles.columns and "low" in candles.columns:
        mask = _spike_candle_mask(candles, atr_window, spike_atr_mult)

    outcome_up, per_share_pnl = score_resolution(
        entry_close=candles["close"],
        next_high=next_high,
        next_low=next_low,
        next_close=next_close,
        signals=sig,
        mode=resolution_mode,
        win_payout=0.95,
        buy_price=0.50,
        half_spread=half_spread,
        spike_candle_mask=mask,
    )

    active = (sig != 0) & outcome_up.notna()
    wins = per_share_pnl > 0
    trade_pnl = (per_share_pnl * size).where(active, 0.0)

    trade_count = int(active.sum())
    returns = trade_pnl.loc[active]
    win_rate = float(wins.loc[active].mean()) if trade_count else 0.0
    total_pnl = float(trade_pnl.sum())
    sharpe = (
        float((returns.mean() / returns.std(ddof=0)) * np.sqrt(len(returns)))
        if trade_count and returns.std(ddof=0) > 0
        else 0.0
    )
    equity = trade_pnl.cumsum()
    running_max = equity.cummax()
    drawdown = float((equity - running_max).min()) if not equity.empty else 0.0

    return {
        "sharpe": sharpe,
        "pnl": total_pnl,
        "win_rate": win_rate,
        "trades": trade_count,
        "drawdown": drawdown,
    }


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #


def fmt_row(label: str, te_m: dict, tr_m: dict) -> str:
    wr_color = "\033[32m" if te_m["win_rate"] >= 0.53 else ("\033[31m" if te_m["win_rate"] < 0.50 else "")
    reset = "\033[0m" if wr_color else ""
    pnl_sign = "+" if te_m["pnl"] >= 0 else ""
    return (
        f"{label:<52}  {wr_color}{te_m['win_rate']:>6.1%}{reset}  "
        f"{pnl_sign}{te_m['pnl']:>9.1f}  {te_m['drawdown']:>8.1f}  "
        f"{tr_m['sharpe']:>7.2f}  {te_m['sharpe']:>7.2f}  {te_m['trades']:>7}"
    )


def print_leaderboard(results: list[dict], title: str, n: int = 20) -> None:
    width = 110
    header = f"\n{'Config':<52}  {'Win%':>6}  {'TestPnL':>9}  {'MaxDD':>8}  {'TrSh':>7}  {'TeSh':>7}  {'Trades':>7}"
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(header)
    print("-" * width)
    for r in results[:n]:
        print(fmt_row(r["label"], r["test"], r["train"]))
    print("=" * width)


# --------------------------------------------------------------------------- #
# Core sweep
# --------------------------------------------------------------------------- #


def run_sweep(
    asset: str,
    tf: str,
    use_ci_sizing: bool,
    _score: Callable[..., dict[str, float]],
) -> list[dict]:
    """Run the full adx param sweep for one asset/timeframe.

    Returns a list of result dicts sorted by test Sharpe (descending).
    Selection discipline: all results are reported; caller filters by train Sharpe
    if needed for param selection.
    """
    print(f"\n[sweep] Loading {asset}/{tf}...")
    candles = load_candles(asset, tf)
    train, test = split_by_date(candles, CUTOFF)
    print(f"  train: {len(train):,} candles ({train.index[0].date()} – {train.index[-1].date()})")
    print(f"  test:  {len(test):,} candles ({test.index[0].date()} – {test.index[-1].date()})")

    strategy = StreakADXStrategy(asset=asset.upper())

    combos = list(product(TRIGGER_GRID, ADX_PERIOD_GRID, ADX_THRESHOLD_GRID, EMA_PERIOD_GRID))
    print(f"  sweeping {len(combos)} combos (trigger × adx_period × adx_threshold × ema_period)...")

    results: list[dict] = []
    for trigger, adx_period, adx_threshold, ema_period in combos:
        strat_params = {
            "trigger": trigger,
            "adx_period": adx_period,
            "adx_threshold": adx_threshold,
            "size": 15.0,
            "use_ci_sizing": use_ci_sizing,
        }
        gate = TrendFilter(ema_period=ema_period, mode="veto_with_trend")

        train_sigs = gate.apply(strategy.evaluate(train, **strat_params), train)
        test_sigs = gate.apply(strategy.evaluate(test, **strat_params), test)

        tr_m = _score(train_sigs, train)
        te_m = _score(test_sigs, test)

        label = f"trig={trigger} adx_p={adx_period:>2} adx_thr={adx_threshold:>2} ema={ema_period:>3}"
        results.append(
            {
                "label": label,
                "asset": asset,
                "tf": tf,
                "trigger": trigger,
                "adx_period": adx_period,
                "adx_threshold": adx_threshold,
                "ema_period": ema_period,
                "use_ci_sizing": use_ci_sizing,
                "train": tr_m,
                "test": te_m,
            }
        )

    # Sort by test Sharpe for display; train Sharpe used for selection discipline
    results.sort(key=lambda x: x["test"]["sharpe"], reverse=True)
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep StreakADX params with TrendFilter")
    parser.add_argument("--asset", default=None, help="Single asset (e.g. eth). Default: all three targets.")
    parser.add_argument("--tf", default=None, help="Single timeframe (e.g. 5m). Default: all three targets.")
    parser.add_argument(
        "--resolution-mode",
        choices=["close", "intrabar_conservative"],
        default="close",
    )
    parser.add_argument(
        "--half-spread",
        type=float,
        default=0.005,
        help="Half bid-ask spread (default: 0.005)",
    )
    parser.add_argument(
        "--atr-window",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--spike-atr-mult",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--no-ci-sizing",
        action="store_true",
        help="Disable CI-based bet sizing (use_ci_sizing=False). Matches live bot behaviour.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top results to print per asset/tf (default: 20)",
    )
    args = parser.parse_args()

    use_ci_sizing = not args.no_ci_sizing

    _score = functools.partial(
        score_signals,
        resolution_mode=args.resolution_mode,
        half_spread=args.half_spread,
        atr_window=args.atr_window,
        spike_atr_mult=args.spike_atr_mult,
    )

    print(f"resolution_mode={args.resolution_mode}  half_spread={args.half_spread}  use_ci_sizing={use_ci_sizing}")
    print(f"gate: TrendFilter(mode=veto_with_trend, ema in {EMA_PERIOD_GRID})")
    print(
        f"strategy: StreakADX  trigger∈{TRIGGER_GRID}  adx_period∈{ADX_PERIOD_GRID}  adx_threshold∈{ADX_THRESHOLD_GRID}"
    )

    # Determine which targets to run
    if args.asset and args.tf:
        targets = [(args.asset, args.tf)]
    elif args.asset:
        targets = [(args.asset, tf) for _, tf in DEFAULT_TARGETS if args.asset in (a for a, _ in DEFAULT_TARGETS)]
        # just filter by asset
        targets = [(a, tf) for a, tf in DEFAULT_TARGETS if a == args.asset]
    elif args.tf:
        targets = [(a, tf) for a, tf in DEFAULT_TARGETS if tf == args.tf]
    else:
        targets = DEFAULT_TARGETS

    OUT_DIR.mkdir(exist_ok=True)
    all_results: list[dict] = []

    for asset, tf in targets:
        results = run_sweep(asset, tf, use_ci_sizing, _score)
        all_results.extend(results)

        print_leaderboard(results, f"TOP {args.top} — {asset.upper()}/{tf}", n=args.top)

        # Best by train Sharpe (selection-safe)
        best_train = max(results, key=lambda x: x["train"]["sharpe"])
        print(
            f"\n  ► Best by TRAIN Sharpe: {best_train['label']}"
            f"  train={best_train['train']['sharpe']:.2f}"
            f"  test={best_train['test']['sharpe']:.2f}"
        )

        # Save per-asset/tf JSON
        out_path = OUT_DIR / f"adx_sweep_{asset}_{tf}.json"
        out_path.write_text(
            json.dumps(
                [
                    {
                        "label": r["label"],
                        "params": {
                            k: r[k] for k in ("trigger", "adx_period", "adx_threshold", "ema_period", "use_ci_sizing")
                        },
                        "train": r["train"],
                        "test": r["test"],
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        print(f"  Saved → {out_path}")

    # Cross-asset summary: top-5 per asset/tf sorted by test Sharpe
    if len(targets) > 1:
        width = 110
        print("\n" + "=" * width)
        print("  CROSS-ASSET SUMMARY — top-5 per target, sorted by test Sharpe")
        print("=" * width)
        header = f"  {'Asset/TF':<10}  {'Config':<52}  {'Win%':>6}  {'TeSh':>7}  {'TrSh':>7}  {'Trades':>7}"
        print(header)
        print("-" * width)
        for asset, tf in targets:
            target_results = [r for r in all_results if r["asset"] == asset and r["tf"] == tf]
            for r in target_results[:5]:
                te = r["test"]
                tr = r["train"]
                wr_color = "\033[32m" if te["win_rate"] >= 0.53 else ("\033[31m" if te["win_rate"] < 0.50 else "")
                reset = "\033[0m" if wr_color else ""
                print(
                    f"  {asset}/{tf:<8}  {r['label']:<52}  "
                    f"{wr_color}{te['win_rate']:>6.1%}{reset}  "
                    f"{te['sharpe']:>7.2f}  {tr['sharpe']:>7.2f}  {te['trades']:>7}"
                )
            print()
        print("=" * width)

    print(
        "\nNext step: pick the best config by TRAIN Sharpe (not test Sharpe) to avoid overfitting.\n"
        "Then re-run the gate sweep (sweep_gates.py --strategy streak_adx --walk-forward)\n"
        "with those locked params to verify the result holds before re-launching the bot.\n"
    )


if __name__ == "__main__":
    main()
