"""Sweep all gate combinations on the best strategy (streak_reversal / eth / 5m).

Usage:
    uv run python scripts/sweep_gates.py
    uv run python scripts/sweep_gates.py --asset btc --tf 15m
    uv run python scripts/sweep_gates.py --sort sharpe

For each gate configuration the script:
  1. Generates base signals (strategy with locked best params)
  2. Applies gate(s)
  3. Evaluates on TRAINING set to find best config per gate class
  4. Validates the best config of each gate class on the held-out TEST set
  5. Prints a sorted leaderboard comparing baseline vs all gate configs

The locked strategy params come from the existing sweep CSV so we never
look at test data during gate selection (no data leakage).
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from polymarket_algo.backtest.engine import _spike_candle_mask, parameter_sweep, score_resolution, walk_forward_split
from polymarket_algo.strategies.funding_rate_extremes import FundingRateFilter
from polymarket_algo.strategies.gates import TrendFilter, VolAccelGate, VolatilityGate, VolumeFilter
from polymarket_algo.strategies.session_filter import SessionFilter
from polymarket_algo.strategies.streak_adx import StreakADXStrategy
from polymarket_algo.strategies.streak_reversal import StreakReversalStrategy
from polymarket_algo.strategies.streak_rsi import StreakRSIStrategy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "backtest_results"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_candles(asset: str, tf: str) -> pd.DataFrame:
    path = DATA_DIR / f"{asset}_{tf}.parquet"
    df = pd.read_parquet(path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
    return df.sort_index()


def split_by_date(candles: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split candles at a date boundary (e.g. '2024-01-01')."""
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    train = candles.loc[candles.index < cutoff_ts].copy()
    test = candles.loc[candles.index >= cutoff_ts].copy()
    return train, test


def load_best_params(asset: str, tf: str, strategy_name: str = "streak_reversal") -> dict:
    """Load best params from the existing sweep CSV (no test leakage)."""
    sweep_path = OUT_DIR / f"{strategy_name}_{asset}_{tf}" / "sweep.csv"
    if sweep_path.exists():
        df = pd.read_csv(sweep_path)
        row = df.iloc[0].to_dict()
        # Extract only param columns (drop metric columns)
        metric_cols = {"win_rate", "total_pnl", "max_drawdown", "sharpe_ratio", "trade_count"}
        return {k: v for k, v in row.items() if k not in metric_cols}
    # Fallback defaults
    return {"trigger": 5, "size": 20.0}


def score_signals(
    signals: pd.DataFrame,
    candles: pd.DataFrame,
    win_payout: float = 0.95,
    buy_price: float = 0.50,
    resolution_mode: str = "close",
    half_spread: float = 0.0,
    atr_window: int = 20,
    spike_atr_mult: float = 2.5,
) -> dict[str, float]:
    """Compute metrics for a gated signal DataFrame against candle outcomes.

    Parameters
    ----------
    resolution_mode:
        Passed through to ``score_resolution``.  Use ``"intrabar_conservative"``
        to apply a haircut on spike candles only (better Polymarket approximation).
    atr_window, spike_atr_mult:
        Spike-candle detection parameters.  A resolution candle is flagged as
        a spike when its range exceeds ``spike_atr_mult × rolling_atr`` (window
        = ``atr_window`` bars, no lookahead).  Only used in
        ``"intrabar_conservative"`` mode.
    """
    sig = signals["signal"].astype(int)
    size = signals["size"].astype(float) if "size" in signals.columns else pd.Series(15.0, index=signals.index)

    next_close = candles["close"].shift(-1)
    next_high = candles["high"].shift(-1) if "high" in candles.columns else next_close
    next_low = candles["low"].shift(-1) if "low" in candles.columns else next_close

    mask: pd.Series | None = None
    if resolution_mode == "intrabar_conservative" and "high" in candles.columns and "low" in candles.columns:
        mask = _spike_candle_mask(candles, atr_window, spike_atr_mult)

    outcome_up, per_share_pnl = score_resolution(
        entry_close=candles["close"],
        next_high=next_high,
        next_low=next_low,
        next_close=next_close,
        signals=sig,
        mode=resolution_mode,
        win_payout=win_payout,
        buy_price=buy_price,
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


def apply_gates(base_signals: pd.DataFrame, candles: pd.DataFrame, gates: list) -> pd.DataFrame:
    """Apply an ordered list of gate objects to a base signals DataFrame."""
    signals = base_signals.copy()
    for gate in gates:
        signals = gate.apply(signals, candles)
    return signals


def fmt_row(label: str, m: dict, tr_m: dict | None = None) -> str:
    wr_color = "\033[32m" if m["win_rate"] >= 0.53 else ("\033[31m" if m["win_rate"] < 0.50 else "")
    reset = "\033[0m" if wr_color else ""
    pnl_sign = "+" if m["pnl"] >= 0 else ""
    tr_sharpe_str = f"{tr_m['sharpe']:>7.2f}" if tr_m is not None else "       "
    return (
        f"{label:<42}  {wr_color}{m['win_rate']:>6.1%}{reset}  "
        f"{pnl_sign}{m['pnl']:>8.2f}  {m['drawdown']:>8.2f}  "
        f"{tr_sharpe_str}  {m['sharpe']:>7.2f}  {m['trades']:>6}"
    )


# --------------------------------------------------------------------------- #
# Gate grid definitions
# --------------------------------------------------------------------------- #

SESSION_GRIDS: list[tuple[str, list[tuple[int, int]] | None]] = [
    ("session:all_day", None),
    ("session:london+us", [(7, 21)]),
    ("session:us_only", [(13, 21)]),
    ("session:london_only", [(7, 16)]),
    ("session:asian_only", [(0, 8)]),
    ("session:peak_overlap", [(13, 16)]),
    ("session:no_asian", [(7, 22)]),
]

VOLATILITY_GRIDS = list(
    product(
        [0.05, 0.10, 0.20],  # low_pct
        [0.80, 0.90, 0.95],  # high_pct
        [144, 288, 576],  # window (1h, 1d, 2d of 5m bars)
    )
)

VOLUME_GRIDS = list(
    product(
        [0.5, 0.75, 1.0, 1.25, 1.5],  # min_mult
        [48, 96, 288],  # window
    )
)

TREND_GRIDS = list(
    product(
        [50, 100, 200],  # ema_period
        ["veto_counter_trend", "veto_with_trend"],  # mode
    )
)

FUNDING_GRIDS = [1.0, 1.5, 2.0, 2.5, 3.0]  # z_threshold

VOL_ACCEL_GRIDS = list(
    product(
        [1.5, 2.0, 2.5, 3.0],  # threshold
        [1.5, 2.0, 2.5],  # boost_factor
        [3, 6, 12],  # short_window
    )
)
# Must match streak_bot.py --vol-accel-long-window default (240 = 20h at 5m).
# Using 288 here while live fetches only ~262 bars caused a silent mismatch.
VOL_ACCEL_LONG_WINDOW = 240


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep gate combinations on the best strategy")
    parser.add_argument("--asset", default="eth")
    parser.add_argument("--tf", default="5m")
    parser.add_argument("--strategy", default="streak_reversal")
    parser.add_argument("--sort", choices=["sharpe", "pnl", "win_rate", "trades"], default="sharpe")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Sweep strategy params on train period only (no data leakage)",
    )
    parser.add_argument(
        "--cutoff",
        default="2024-01-01",
        help="Date cutoff for walk-forward split (default: 2024-01-01)",
    )
    parser.add_argument(
        "--resolution-mode",
        choices=["close", "intrabar_conservative"],
        default="close",
        help=(
            "Win condition model. 'close' (default): next_close > entry_close. "
            "'intrabar_conservative': also marks wins as losses when the intrabar "
            "reversal shadow exceeds the net directional move — a tighter proxy for "
            "Polymarket oracle resolution during vol-spike regimes."
        ),
    )
    parser.add_argument(
        "--half-spread",
        type=float,
        default=0.005,
        help="Half bid-ask spread deducted from entry (price units, e.g. 0.005)",
    )
    parser.add_argument(
        "--atr-window",
        type=int,
        default=20,
        help="Rolling ATR window (bars) for spike-candle detection (default: 20 = 100 min at 5m).",
    )
    parser.add_argument(
        "--spike-atr-mult",
        type=float,
        default=2.5,
        help=(
            "A resolution candle is flagged as a spike when its range exceeds "
            "spike_atr_mult × rolling_atr.  Default 2.5.  Only used with "
            "--resolution-mode intrabar_conservative."
        ),
    )
    args = parser.parse_args()

    # Bind resolution mode and scoring params for all score_signals calls in this run.
    import functools

    _score = functools.partial(
        score_signals,
        resolution_mode=args.resolution_mode,
        half_spread=args.half_spread,
        atr_window=args.atr_window,
        spike_atr_mult=args.spike_atr_mult,
    )
    if args.resolution_mode != "close":
        print(f"[resolution] mode={args.resolution_mode}  atr={args.atr_window}  spike_mult={args.spike_atr_mult}")
    if args.half_spread != 0.0:
        print(f"[spread] half_spread={args.half_spread}")

    candles = load_candles(args.asset, args.tf)

    if args.walk_forward:
        train, test = split_by_date(candles, args.cutoff)
    else:
        train, test = walk_forward_split(candles, train_ratio=0.75)

    if args.strategy == "streak_rsi":
        strategy = StreakRSIStrategy(asset=args.asset.upper())
    elif args.strategy == "streak_adx":
        strategy = StreakADXStrategy(asset=args.asset.upper())
    else:
        strategy = StreakReversalStrategy()

    if args.walk_forward:
        print(f"\nWalk-forward mode: sweeping strategy params on TRAIN only (cutoff: {args.cutoff})")
        sweep_df = parameter_sweep(
            train,
            strategy,
            strategy.param_grid,
            resolution_mode=args.resolution_mode,
            half_spread=args.half_spread,
            atr_window=args.atr_window,
            spike_atr_mult=args.spike_atr_mult,
        )
        sweep_df = sweep_df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
        metric_cols = {"win_rate", "total_pnl", "max_drawdown", "sharpe_ratio", "trade_count"}
        best_params = {k: sweep_df.iloc[0][k] for k in sweep_df.columns if k not in metric_cols}
        print(f"Best train params: {best_params}  (train Sharpe: {sweep_df.iloc[0]['sharpe_ratio']:.2f})")
    else:
        best_params = load_best_params(args.asset, args.tf, args.strategy)

    print(f"\nGate sweep: {args.strategy} / {args.asset} / {args.tf}")
    print(f"Best params: {best_params}")
    print(f"Train: {len(train):,} candles  |  Test: {len(test):,} candles\n")

    # Pre-generate base signals (locked params, no gate)
    train_base = strategy.evaluate(train, **best_params)
    test_base = strategy.evaluate(test, **best_params)

    # Baseline scores
    baseline_train = _score(train_base, train)
    baseline_test = _score(test_base, test)

    # Results accumulator: list of (label, train_metrics, test_metrics, gates_list)
    results: list[tuple[str, dict, dict, list]] = []
    results.append(("baseline (no gate)", baseline_train, baseline_test, []))

    # --- Session filter sweep ---
    for label, hours in SESSION_GRIDS:
        if hours is None:
            continue  # skip all_day (same as baseline)
        gate = SessionFilter(allowed_hours=hours)
        tr_m = _score(apply_gates(train_base, train, [gate]), train)
        te_m = _score(apply_gates(test_base, test, [gate]), test)
        results.append((label, tr_m, te_m, [gate]))

    # --- Volatility gate sweep — pick best on train ---
    vol_best_train_sharpe = -999.0
    vol_best: tuple | None = None
    for low_pct, high_pct, window in VOLATILITY_GRIDS:
        gate = VolatilityGate(low_pct=low_pct, high_pct=high_pct, window=window)
        tr_m = _score(apply_gates(train_base, train, [gate]), train)
        if tr_m["sharpe"] > vol_best_train_sharpe:
            vol_best_train_sharpe = tr_m["sharpe"]
            vol_best = (low_pct, high_pct, window, gate, tr_m)
    if vol_best:
        low_pct, high_pct, window, gate, tr_m = vol_best
        te_m = _score(apply_gates(test_base, test, [gate]), test)
        label = f"vol_gate:low={low_pct:.0%} high={high_pct:.0%} w={window}"
        results.append((label, tr_m, te_m, [gate]))
        print(f"[vol] best train config: {label}  sharpe={tr_m['sharpe']:.2f}")

    # --- Volume filter sweep — pick best on train ---
    vm_best_train_sharpe = -999.0
    vm_best: tuple | None = None
    for min_mult, window in VOLUME_GRIDS:
        gate = VolumeFilter(window=window, min_mult=min_mult)
        tr_m = _score(apply_gates(train_base, train, [gate]), train)
        if tr_m["sharpe"] > vm_best_train_sharpe:
            vm_best_train_sharpe = tr_m["sharpe"]
            vm_best = (min_mult, window, gate, tr_m)
    if vm_best:
        min_mult, window, gate, tr_m = vm_best
        te_m = _score(apply_gates(test_base, test, [gate]), test)
        label = f"vol_filter:mult={min_mult:.2f} w={window}"
        results.append((label, tr_m, te_m, [gate]))
        print(f"[volume] best train config: {label}  sharpe={tr_m['sharpe']:.2f}")

    # --- Trend filter sweep — all combos reported individually ---
    tf_best_train_sharpe = -999.0
    tf_best: tuple | None = None
    for ema_period, mode in TREND_GRIDS:
        gate = TrendFilter(ema_period=ema_period, mode=mode)
        tr_m = _score(apply_gates(train_base, train, [gate]), train)
        te_m = _score(apply_gates(test_base, test, [gate]), test)
        label = f"trend:ema={ema_period} mode={mode}"
        results.append((label, tr_m, te_m, [gate]))
        if tr_m["sharpe"] > tf_best_train_sharpe:
            tf_best_train_sharpe = tr_m["sharpe"]
            tf_best = (ema_period, mode, gate, tr_m)
    if tf_best:
        print(f"[trend] best train config: trend:ema={tf_best[0]} mode={tf_best[1]}  sharpe={tf_best[3]['sharpe']:.2f}")

    # --- Funding rate filter sweep ---
    fr_best_train_sharpe = -999.0
    fr_best: tuple | None = None
    for z_threshold in FUNDING_GRIDS:
        gate = FundingRateFilter(z_threshold=z_threshold)
        tr_m = _score(apply_gates(train_base, train, [gate]), train)
        if tr_m["sharpe"] > fr_best_train_sharpe:
            fr_best_train_sharpe = tr_m["sharpe"]
            fr_best = (z_threshold, gate, tr_m)
    if fr_best:
        z_threshold, gate, tr_m = fr_best
        te_m = _score(apply_gates(test_base, test, [gate]), test)
        label = f"funding_filter:z={z_threshold:.1f}"
        results.append((label, tr_m, te_m, [gate]))
        print(f"[funding] best train config: {label}  sharpe={tr_m['sharpe']:.2f}")

    # --- Vol accel gate sweep — pick best on train ---
    va_best_train_sharpe = -999.0
    va_best: tuple | None = None
    for threshold, boost_factor, short_window in VOL_ACCEL_GRIDS:
        gate = VolAccelGate(
            short_window=short_window,
            long_window=VOL_ACCEL_LONG_WINDOW,
            threshold=threshold,
            boost_factor=boost_factor,
        )
        tr_m = _score(apply_gates(train_base, train, [gate]), train)
        if tr_m["sharpe"] > va_best_train_sharpe:
            va_best_train_sharpe = tr_m["sharpe"]
            va_best = (threshold, boost_factor, short_window, gate, tr_m)
    if va_best:
        threshold, boost_factor, short_window, gate, tr_m = va_best
        te_m = _score(apply_gates(test_base, test, [gate]), test)
        label = f"vol_accel:thr={threshold:.1f} boost={boost_factor:.1f} sw={short_window}"
        results.append((label, tr_m, te_m, [gate]))
        print(f"[vol_accel] best train config: {label}  sharpe={tr_m['sharpe']:.2f}")

    # --- Best 2-gate combos: pair each top single-gate with session filters ---
    # Find the top 3 single-gate configs by training sharpe (excluding baseline)
    single_results = [
        (label, tr_m, te_m, gates) for label, tr_m, te_m, gates in results if label != "baseline (no gate)"
    ]
    single_results.sort(key=lambda x: x[1]["sharpe"], reverse=True)
    top_singles = single_results[:3]

    for s_label, _, _, s_gates in top_singles:
        for sess_label, hours in SESSION_GRIDS[1:]:  # skip all_day
            combo_gates = s_gates + [SessionFilter(allowed_hours=hours)]
            tr_m = _score(apply_gates(train_base, train, combo_gates), train)
            te_m = _score(apply_gates(test_base, test, combo_gates), test)
            label = f"{s_label} + {sess_label}"
            results.append((label, tr_m, te_m, combo_gates))

    # --- VolAccelGate + TrendFilter combos ---
    if va_best and tf_best:
        _, _, _, va_gate, _ = va_best
        _, _, tf_gate, _ = tf_best
        combo_gates = [tf_gate, va_gate]
        tr_m = _score(apply_gates(train_base, train, combo_gates), train)
        te_m = _score(apply_gates(test_base, test, combo_gates), test)
        label = f"trend:ema={tf_best[0]} mode={tf_best[1]} + vol_accel:thr={va_best[0]:.1f} boost={va_best[1]:.1f}"
        results.append((label, tr_m, te_m, combo_gates))
        print(f"[combo] trend+vol_accel: train sharpe={tr_m['sharpe']:.2f}")

    # --- Print leaderboard ---
    sort_key = {
        "sharpe": lambda x: x[2]["sharpe"],
        "pnl": lambda x: x[2]["pnl"],
        "win_rate": lambda x: x[2]["win_rate"],
        "trades": lambda x: x[2]["trades"],
    }[args.sort]
    results.sort(key=sort_key, reverse=True)

    header = (
        f"\n{'Gate config':<42}  {'Win%':>6}  {'Test PnL':>8}  {'Drawdn':>8}  {'TrSh':>7}  {'TeSh':>7}  {'Trades':>6}"
    )
    width = 95
    print("\n" + "=" * width)
    print(f"TEST SET RESULTS — sorted by {args.sort}")
    print("=" * width)
    print(header)
    print("-" * width)
    for label, tr_m, te_m, _ in results:
        marker = " ◄ baseline" if label == "baseline (no gate)" else ""
        print(fmt_row(label, te_m, tr_m) + marker)
    print("=" * width)

    # --- Save to JSON ---
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"gate_sweep_{args.strategy}_{args.asset}_{args.tf}.json"
    saved = [{"label": label, "train": tr_m, "test": te_m} for label, tr_m, te_m, _ in results]
    out_path.write_text(json.dumps(saved, indent=2))
    print(f"\nSaved {len(results)} results → {out_path}\n")


if __name__ == "__main__":
    main()
