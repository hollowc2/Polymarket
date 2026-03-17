"""APEX strategy parameter sweep and walk-forward backtest.

Uses Binance kline parquet files already on disk — no live API calls.
CVD features (buy_vol, sell_vol, delta) are derived from the
taker_buy_base_asset_volume column present in every parquet file.

Liquidation and funding enrichment require live Binance API calls
and are attempted optionally (skipped gracefully on failure).

Usage
-----
    # Quick sweep — ETH 5m, no enrichment (plain CVD from parquet)
    uv run python scripts/backtest_apex.py

    # All assets, both timeframes
    uv run python scripts/backtest_apex.py --asset all --tf all

    # Full enrichment (adds liq + funding from Binance API)
    uv run python scripts/backtest_apex.py --enrich

    # Walk-forward (pre-2024 train, 2024+ test)
    uv run python scripts/backtest_apex.py --walk-forward --cutoff 2024-01-01
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from polymarket_algo.strategies.apex_hybrid import ApexHybridStrategy
from polymarket_algo.strategies.apex_strategy import ApexStrategy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "backtest_results" / "apex"

ASSETS = ["btc", "eth", "sol", "xrp"]
TFS = ["5m", "15m"]

ASSET_TO_SYMBOL = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_enrich(
    asset: str,
    tf: str,
    enrich_liq: bool = False,
    enrich_funding: bool = False,
) -> pd.DataFrame:
    """Load parquet and derive CVD features from taker_buy columns.

    Binance kline columns used:
        taker_buy_base_asset_volume → buy_vol
        volume - taker_buy_base_asset_volume → sell_vol
        buy_vol - sell_vol → delta (per-candle CVD increment)

    Liquidation and funding enrichment are attempted only when
    enrich_liq / enrich_funding are True.
    """
    path = DATA_DIR / f"{asset}_{tf}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No parquet at {path}")

    df = pd.read_parquet(path)

    # Normalise index
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
    df = df.sort_index()

    # Core OHLCV
    ohlcv = df[["open", "high", "low", "close", "volume"]].copy()

    # CVD from taker columns (always available in Binance klines)
    if "taker_buy_base_asset_volume" in df.columns:
        buy_vol = df["taker_buy_base_asset_volume"].astype(float)
        sell_vol = df["volume"].astype(float) - buy_vol
        ohlcv["buy_vol"] = buy_vol
        ohlcv["sell_vol"] = sell_vol.clip(lower=0.0)
        ohlcv["delta"] = ohlcv["buy_vol"] - ohlcv["sell_vol"]
        ohlcv["cvd"] = ohlcv["delta"].cumsum()

    # Optional enrichment via live Binance API
    symbol = ASSET_TO_SYMBOL[asset]
    if enrich_liq or enrich_funding:
        try:
            from polymarket_algo.data.enrich import enrich_candles

            enriched = enrich_candles(
                ohlcv,
                symbol=symbol,
                include_cvd=False,  # already computed above
                include_liq=enrich_liq,
                include_funding=enrich_funding,
            )
            ohlcv = enriched
        except Exception as exc:
            print(f"  [enrich] Skipped for {asset}/{tf}: {exc}")

    return ohlcv


def split_candles(
    candles: pd.DataFrame,
    mode: str,
    cutoff: str,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "walk_forward":
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
        train = candles[candles.index < cutoff_ts]
        test = candles[candles.index >= cutoff_ts]
    else:
        n = len(candles)
        split = int(n * train_ratio)
        train = candles.iloc[:split]
        test = candles.iloc[split:]
    return train.copy(), test.copy()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

WIN_PAYOUT = 0.95  # Polymarket pays ~95 cents on $1 winner (after fees)
BUY_PRICE = 0.50  # Assumes symmetric 50/50 market pricing


def score_signals(signals: pd.DataFrame, candles: pd.DataFrame) -> dict:
    """Score strategy signals against next-candle close direction.

    Win condition:
        signal=+1 and next close > current close  → UP win
        signal=-1 and next close < current close  → DOWN win
    """
    sig = signals["signal"].astype(int)
    size = signals["size"].astype(float) if "size" in signals.columns else pd.Series(15.0, index=signals.index)

    next_close = candles["close"].shift(-1)
    outcome_up = (next_close > candles["close"]).astype(int)

    active = (sig != 0) & outcome_up.notna()
    wins = ((sig == 1) & (outcome_up == 1)) | ((sig == -1) & (outcome_up == 0))

    per_share_pnl = np.where(wins, WIN_PAYOUT - BUY_PRICE, -BUY_PRICE)
    per_share_pnl = pd.Series(per_share_pnl, index=candles.index)
    trade_pnl = (per_share_pnl * size).where(active, 0.0)

    trade_count = int(active.sum())
    returns = trade_pnl.loc[active]
    win_rate = float(wins.loc[active].mean()) if trade_count else 0.0
    total_pnl = float(trade_pnl.sum())
    sharpe = (
        float((returns.mean() / returns.std(ddof=0)) * np.sqrt(len(returns)))
        if trade_count > 1 and returns.std(ddof=0) > 0
        else 0.0
    )
    equity = trade_pnl.cumsum()
    drawdown = float((equity - equity.cummax()).min()) if not equity.empty else 0.0

    return {
        "sharpe": round(sharpe, 4),
        "pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 4),
        "trades": trade_count,
        "drawdown": round(drawdown, 2),
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

# Focused sweep grid — 5 params × ~3 values each = manageable runtime
APEX_SWEEP_GRID = {
    "edge_threshold": [0.02, 0.04, 0.06, 0.08],
    "w_obi": [0.4, 0.8, 1.2],
    "w_tfi": [0.4, 0.8, 1.2],
    "w_mp": [0.3, 0.6, 0.9],
    "toxicity_max": [0.55, 0.65, 0.75],
    "kelly_scale": [0.15, 0.25, 0.35],
    "max_bet": [10.0, 15.0, 20.0],
}

# Reduced grid for fast initial runs
APEX_SWEEP_GRID_FAST = {
    "edge_threshold": [0.02, 0.04, 0.06],
    "w_obi": [0.4, 0.8, 1.2],
    "w_tfi": [0.4, 0.8, 1.2],
    "toxicity_max": [0.55, 0.65, 0.75],
    "kelly_scale": [0.20, 0.30],
    "max_bet": [10.0, 15.0],
}

# ApexHybrid sweep grid (separate, matched to its param_grid)
APEX_HYBRID_SWEEP_GRID = {
    "trigger": [3, 4, 5],
    "w_tfi_exhaust": [0.0, 1.0, 2.0],  # 0 = exhaustion off, 1 = default, 2 = aggressive
    "w_mp_exhaust": [0.0, 1.0],  # 0 = off, 1 = on
    "exhaust_mult": [0.5, 1.5],  # how much exhaustion boosts size
    "hawkes_threshold": [0.0, 0.53],  # 0 = gate off, 0.53 = active-market-only
    "kelly_scale": [0.20, 0.30],
    "max_bet": [10.0, 15.0, 20.0],
}
# 3×3×2×2×2×2×3 = 432 combos


def run_param_sweep(
    strategy: ApexStrategy | ApexHybridStrategy,
    train: pd.DataFrame,
    grid: dict,
    top_n: int = 10,
) -> pd.DataFrame:
    """Grid search over APEX params on the training set only.

    Returns a DataFrame sorted by Sharpe, descending.
    """
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(product(*values))

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo, strict=True))
        try:
            signals = strategy.evaluate(train, **params)
            m = score_signals(signals, train)
            rows.append({**params, **m})
        except Exception:
            pass  # skip bad param combos

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("sharpe", ascending=False).reset_index(drop=True).head(top_n * 3)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def color_sharpe(v: float) -> str:
    c = GREEN if v > 1.5 else RED if v < 0 else YELLOW
    return f"{c}{v:>7.2f}{RESET}"


def color_wr(v: float) -> str:
    c = GREEN if v >= 0.53 else RED if v < 0.50 else ""
    r = RESET if c else ""
    return f"{c}{v:>6.1%}{r}"


def print_leaderboard(rows: list[dict], title: str) -> None:
    header = f"\n{'Config':<38}  {'WinR':>6}  {'PnL':>8}  {'DD':>8}  {'TrSh':>6}  {'TeSh':>6}  {'Trades':>6}"
    bar = "=" * 90
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(bar)
    print(header)
    print("-" * 90)
    for r in rows:
        label = r["label"]
        te = r["test"]
        tr = r["train"]
        print(
            f"{label:<38}  {color_wr(te['win_rate'])}  "
            f"{te['pnl']:>+8.2f}  {te['drawdown']:>8.2f}  "
            f"{color_sharpe(tr['sharpe'])}  {color_sharpe(te['sharpe'])}  "
            f"{te['trades']:>6}"
        )
    print(bar + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_asset_tf(
    asset: str,
    tf: str,
    mode: str,
    cutoff: str,
    train_ratio: float,
    enrich_liq: bool,
    enrich_funding: bool,
    fast: bool,
    hybrid: bool = False,
) -> list[dict] | None:
    """Run full APEX (or APEX Hybrid) sweep for one asset/timeframe."""
    print(f"\n{'─' * 60}")
    print(f"  {asset.upper()} / {tf}")
    print(f"{'─' * 60}")

    try:
        t0 = time.time()
        candles = load_and_enrich(asset, tf, enrich_liq=enrich_liq, enrich_funding=enrich_funding)
        check = ["buy_vol", "sell_vol", "delta", "liq_long_usd", "funding_zscore"]
        enriched_cols = [c for c in check if c in candles.columns]
        print(f"  Loaded {len(candles):,} candles | enriched: {enriched_cols}")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
        return None

    train, test = split_candles(candles, mode, cutoff, train_ratio)
    print(f"  Train: {len(train):,} ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"  Test:  {len(test):,} ({test.index[0].date()} → {test.index[-1].date()})")

    if hybrid:
        strategy = ApexHybridStrategy()
        grid = APEX_HYBRID_SWEEP_GRID
    else:
        strategy = ApexStrategy()
        grid = APEX_SWEEP_GRID_FAST if fast else APEX_SWEEP_GRID

    # Parameter sweep on train only
    print(f"  Sweeping {len(list(product(*grid.values()))):,} param combos on train set…")
    sweep_df = run_param_sweep(strategy, train, grid, top_n=20)

    if sweep_df.empty:
        print("  No valid param combos — skipping")
        return None

    metric_cols = {"sharpe", "pnl", "win_rate", "trades", "drawdown"}
    param_cols = [c for c in sweep_df.columns if c not in metric_cols]

    # Evaluate top-20 train configs on test set
    rows = []

    # Baseline: default params
    default_params = {k: strategy.default_params[k] for k in param_cols if k in strategy.default_params}
    base_train = score_signals(strategy.evaluate(train, **default_params), train)
    base_test = score_signals(strategy.evaluate(test, **default_params), test)
    rows.append({"label": "defaults", "train": base_train, "test": base_test, "params": default_params})

    # Top param combos from sweep
    for _, row in sweep_df.iterrows():
        params = {k: row[k] for k in param_cols if k in row}
        try:
            test_sig = strategy.evaluate(test, **params)
            te_m = score_signals(test_sig, test)
            tr_m = {k: row[k] for k in metric_cols if k in row}
            label = (
                f"e={params.get('edge_threshold', 0.04):.2f} "
                f"obi={params.get('w_obi', 0.8):.1f} "
                f"tfi={params.get('w_tfi', 0.8):.1f} "
                f"tox={params.get('toxicity_max', 0.65):.2f} "
                f"ks={params.get('kelly_scale', 0.25):.2f}"
            )
            rows.append({"label": label, "train": tr_m, "test": te_m, "params": params})
        except Exception:
            pass

    # Sort by test Sharpe
    rows.sort(key=lambda r: r["test"]["sharpe"], reverse=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(rows)} configs evaluated")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="APEX backtest + parameter sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--asset",
        default="eth",
        help="Asset to test (btc/eth/sol/xrp/all)",
    )
    parser.add_argument(
        "--tf",
        default="5m",
        help="Timeframe (5m/15m/all)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use date-based split instead of ratio split",
    )
    parser.add_argument(
        "--cutoff",
        default="2024-01-01",
        help="Walk-forward cutoff date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Train fraction for ratio split",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Fetch liquidation + funding from Binance API (slower)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced param grid for quick exploration",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Run ApexHybridStrategy (streak + exhaustion) instead of ApexStrategy",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top configs to show per asset/tf",
    )
    args = parser.parse_args()

    assets = ASSETS if args.asset == "all" else [args.asset]
    tfs = TFS if args.tf == "all" else [args.tf]
    mode = "walk_forward" if args.walk_forward else "ratio"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list] = {}

    for asset in assets:
        for tf in tfs:
            rows = run_asset_tf(
                asset=asset,
                tf=tf,
                mode=mode,
                cutoff=args.cutoff,
                train_ratio=args.train_ratio,
                enrich_liq=args.enrich,
                enrich_funding=args.enrich,
                fast=args.fast,
                hybrid=args.hybrid,
            )
            if rows is None:
                continue

            key = f"{asset}_{tf}"
            all_results[key] = rows

            # Print leaderboard (top N by test Sharpe)
            if mode == "walk_forward":
                split_label = f"walk-forward cutoff={args.cutoff}"
            else:
                split_label = f"ratio={args.train_ratio:.0%}"
            print_leaderboard(
                rows[: args.top],
                title=f"APEX — {asset.upper()} / {tf} — {split_label}",
            )

            # Best config summary
            best = rows[0]
            print(
                f"  Best test Sharpe: {best['test']['sharpe']:.2f} "
                f"| Win: {best['test']['win_rate']:.1%} "
                f"| Trades: {best['test']['trades']} "
                f"| PnL: ${best['test']['pnl']:+.2f}"
            )
            print(f"  Best params: {best['params']}")

            # Save JSON
            prefix = "apex_hybrid" if args.hybrid else "apex"
            out_path = OUT_DIR / f"{prefix}_{key}.json"
            serialisable = [
                {
                    "label": r["label"],
                    "params": r["params"],
                    "train": r["train"],
                    "test": r["test"],
                }
                for r in rows
            ]
            out_path.write_text(json.dumps(serialisable, indent=2))
            print(f"  Saved → {out_path}")

    # Cross-asset summary table
    if len(all_results) > 1:
        print(f"\n{BOLD}{'=' * 60}{RESET}")
        print(f"{BOLD}CROSS-ASSET SUMMARY — best config per pair{RESET}")
        print(f"{'=' * 60}")
        print(f"{'Asset/TF':<14} {'TrainSh':>7} {'TestSh':>7} {'Win%':>6} {'Trades':>7} {'PnL':>8}")
        print("-" * 60)
        for key, rows in all_results.items():
            best = rows[0]
            asset, tf = key.split("_", 1)
            print(
                f"{asset.upper()}/{tf:<9} "
                f"{color_sharpe(best['train']['sharpe'])} "
                f"{color_sharpe(best['test']['sharpe'])} "
                f"{color_wr(best['test']['win_rate'])} "
                f"{best['test']['trades']:>7} "
                f"{best['test']['pnl']:>+8.2f}"
            )
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
