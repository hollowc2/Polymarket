"""ApexML walk-forward training script.

Trains a logistic regression model on multi-TF momentum + microstructure features,
evaluates on a held-out test set, and saves the model to models/.

Walk-forward protocol:
    Train: all data before cutoff (default: 2024-01-01)
    Test:  all data from cutoff onwards

Usage
-----
    # Train ETH 5m (fastest, confirmed best correlation)
    uv run python scripts/train_apex_ml.py --asset eth --tf 5m

    # Train all assets
    uv run python scripts/train_apex_ml.py --asset all --tf 5m

    # Custom cutoff
    uv run python scripts/train_apex_ml.py --asset eth --tf 5m --cutoff 2023-01-01

Expected output (ETH 5m, cutoff 2024-01-01):
    Feature coefficients — expect mom_15m to dominate
    Test Sharpe > 2.0, win rate > 53%
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from polymarket_algo.strategies.apex_ml import ApexMLStrategy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

ASSETS = ["btc", "eth", "sol", "xrp"]
TFS = ["5m", "15m"]

ASSET_TO_SYMBOL = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
}

DEFAULT_CUTOFF = "2024-01-01"
DEFAULT_EDGE_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_candles(asset: str, tf: str) -> pd.DataFrame:
    """Load parquet and derive CVD from taker_buy columns."""
    path = DATA_DIR / f"{asset}_{tf}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No parquet at {path}. Run data fetch first.")

    df = pd.read_parquet(path)

    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
    df = df.sort_index()

    # Core OHLCV
    ohlcv = df[["open", "high", "low", "close", "volume"]].copy()

    # CVD from taker columns (present in Binance klines with VPN)
    if "taker_buy_base_asset_volume" in df.columns:
        buy_vol = df["taker_buy_base_asset_volume"].astype(float)
        sell_vol = df["volume"].astype(float) - buy_vol
        ohlcv["buy_vol"] = buy_vol
        ohlcv["sell_vol"] = sell_vol.clip(lower=0.0)
        ohlcv["delta"] = ohlcv["buy_vol"] - ohlcv["sell_vol"]
        ohlcv["cvd"] = ohlcv["delta"].cumsum()

    # number_of_trades (for trade_count_mtf feature)
    if "number_of_trades" in df.columns:
        ohlcv["number_of_trades"] = df["number_of_trades"].astype(float)

    return ohlcv


# ---------------------------------------------------------------------------
# Backtest evaluation
# ---------------------------------------------------------------------------


def compute_pnl(candles: pd.DataFrame, signals: pd.Series, sizes: pd.Series) -> pd.Series:
    """Compute per-trade PnL from signals and bar-by-bar returns.

    Uses next-bar return to avoid lookahead. A signal of +1 at bar t means we
    bet UP and resolve at bar t+1 close.
    """
    next_ret = candles["close"].pct_change().shift(-1)

    # Win/loss: +1 if direction matches, -1 if not
    won = (signals * np.sign(next_ret)).clip(-1, 1)
    pnl = won * sizes * np.abs(next_ret)

    # Zero out no-signal bars
    pnl = pnl.where(signals != 0, 0.0)
    return pnl.dropna()


def backtest_metrics(pnl: pd.Series, signals: pd.Series) -> dict:
    """Compute Sharpe, win rate, and trade count from per-trade PnL."""
    trades = pnl[signals.reindex(pnl.index).fillna(0) != 0]
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": 0.0, "sharpe": 0.0, "total_pnl": 0.0}

    wins = (trades > 0).sum()
    mean_pnl = trades.mean()
    std_pnl = trades.std()
    sharpe = (mean_pnl / std_pnl * np.sqrt(n)) if std_pnl > 0 else 0.0

    return {
        "n_trades": int(n),
        "win_rate": float(wins / n),
        "sharpe": float(sharpe),
        "total_pnl": float(trades.sum()),
        "avg_pnl": float(mean_pnl),
    }


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_and_evaluate(
    asset: str,
    tf: str,
    cutoff: str,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    save_model: bool = True,
) -> dict:
    """Full walk-forward training pipeline for one asset/TF pair.

    Returns a dict with train/test metrics and feature coefficients.
    """
    print(f"\n{'=' * 60}")
    print(f"  Asset: {asset.upper()}  TF: {tf}  Cutoff: {cutoff}")
    print(f"{'=' * 60}")

    t0 = time.time()
    candles = load_candles(asset, tf)
    print(f"Loaded {len(candles):,} bars  ({candles.index[0].date()} → {candles.index[-1].date()})")

    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    train = candles[candles.index < cutoff_ts]
    test = candles[candles.index >= cutoff_ts]

    if len(train) < 500:
        print(f"  WARNING: Only {len(train)} train bars — skipping")
        return {}
    if len(test) < 200:
        print(f"  WARNING: Only {len(test)} test bars — skipping")
        return {}

    print(f"Train: {len(train):,} bars  Test: {len(test):,} bars")

    # --- Fit model on train split ---
    model = ApexMLStrategy()
    model.fit(train)
    print(f"Training took {time.time() - t0:.1f}s")

    # --- Feature coefficients ---
    importance = model.feature_importance()
    print("\nFeature coefficients (sorted by magnitude):")
    for feat, coef in importance.items():
        bar = "█" * int(abs(coef) * 20)
        sign = "+" if coef >= 0 else "-"
        print(f"  {feat:12s}  {sign}{abs(coef):.4f}  {bar}")

    # --- Evaluate on test split (scaler already fit on train) ---
    # Apply to full candles but only look at test portion signals
    # We need enough warmup bars for multi-TF features (60 bars at 60min = 60+ 5m bars)
    # Use last 200 train bars as warmup
    warmup = candles[candles.index < cutoff_ts].tail(300)
    eval_candles = pd.concat([warmup, test])

    result_all = model.evaluate(eval_candles, edge_threshold=edge_threshold)
    result_test = result_all.loc[test.index]

    # Compute PnL on test set
    test_with_warmup_pnl = pd.Series(0.0, index=test.index)
    for i, ts in enumerate(test.index):
        if i + 1 < len(test.index):
            next_ts = test.index[i + 1]
            signal = result_test.loc[ts, "signal"]
            if signal != 0:
                # Next bar return
                ret = (test.loc[next_ts, "close"] - test.loc[ts, "close"]) / test.loc[ts, "close"]
                won = signal * np.sign(ret) > 0
                size = float(result_test.loc[ts, "size"])
                pnl_val = size * abs(ret) if won else -size * abs(ret)
                test_with_warmup_pnl.iloc[i] = pnl_val

    # Simpler vectorized version
    next_ret = test["close"].pct_change().shift(-1)
    sig = result_test["signal"]
    sz = result_test["size"]
    won_arr = (sig * np.sign(next_ret)).clip(-1, 1)
    pnl_per_bar = (won_arr * sz * next_ret.abs()).fillna(0.0)
    trade_pnl = pnl_per_bar[sig != 0].dropna()

    n_test = len(trade_pnl)
    if n_test == 0:
        print("\nNo trades in test set.")
        return {}

    wins_test = (trade_pnl > 0).sum()
    mean_p = trade_pnl.mean()
    std_p = trade_pnl.std()
    test_sharpe = float(mean_p / std_p * np.sqrt(n_test)) if std_p > 0 else 0.0
    test_wr = float(wins_test / n_test)

    print(f"\nTest set results (edge_threshold={edge_threshold}):")
    print(f"  Trades   : {n_test:,}")
    print(f"  Win rate : {test_wr:.1%}")
    print(f"  Sharpe   : {test_sharpe:.2f}")
    print(f"  Total PnL: ${trade_pnl.sum():.2f}")
    print(f"  Avg PnL  : ${mean_p:.4f}/trade")

    # --- Train-set eval (sanity check for overfitting) ---
    result_train = model.evaluate(train, edge_threshold=edge_threshold)
    next_ret_tr = train["close"].pct_change().shift(-1)
    sig_tr = result_train["signal"]
    sz_tr = result_train["size"]
    won_tr = (sig_tr * np.sign(next_ret_tr)).clip(-1, 1)
    pnl_tr = (won_tr * sz_tr * next_ret_tr.abs()).fillna(0.0)
    trade_pnl_tr = pnl_tr[sig_tr != 0].dropna()
    n_tr = len(trade_pnl_tr)
    tr_std = trade_pnl_tr.std()
    tr_sharpe = float(trade_pnl_tr.mean() / tr_std * np.sqrt(n_tr)) if (n_tr > 0 and tr_std > 0) else 0.0
    tr_wr = float((trade_pnl_tr > 0).sum() / n_tr) if n_tr > 0 else 0.0
    print("\nTrain-set (sanity — expect similar to test if no overfit):")
    print(f"  Trades   : {n_tr:,}")
    print(f"  Win rate : {tr_wr:.1%}")
    print(f"  Sharpe   : {tr_sharpe:.2f}")

    # --- Save model ---
    model_path = MODELS_DIR / f"apex_ml_{asset}_{tf}.json"
    if save_model:
        model.save(model_path)
        print(f"\nModel saved → {model_path}")

    result = {
        "asset": asset,
        "tf": tf,
        "cutoff": cutoff,
        "train_bars": len(train),
        "test_bars": len(test),
        "test_trades": n_test,
        "test_win_rate": test_wr,
        "test_sharpe": test_sharpe,
        "test_total_pnl": float(trade_pnl.sum()),
        "train_trades": n_tr,
        "train_win_rate": tr_wr,
        "train_sharpe": tr_sharpe,
        "feature_coefs": {k: round(v, 6) for k, v in importance.items()},
        "model_path": str(model_path),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ApexML walk-forward logistic regression training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--asset",
        choices=["btc", "eth", "sol", "xrp", "all"],
        default="eth",
        help="Asset to train on (all = btc+eth+sol+xrp)",
    )
    parser.add_argument(
        "--tf",
        choices=["5m", "15m", "all"],
        default="5m",
        help="Timeframe (all = 5m+15m)",
    )
    parser.add_argument(
        "--cutoff",
        default=DEFAULT_CUTOFF,
        metavar="DATE",
        help="Walk-forward cutoff (train before, test after)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=DEFAULT_EDGE_THRESHOLD,
        metavar="FLOAT",
        help="Minimum |edge| = |P(UP) - 0.5| to generate a signal",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save model files (dry-run evaluation only)",
    )
    args = parser.parse_args()

    assets = ASSETS if args.asset == "all" else [args.asset]
    tfs = TFS if args.tf == "all" else [args.tf]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for asset in assets:
        for tf in tfs:
            try:
                result = train_and_evaluate(
                    asset=asset,
                    tf=tf,
                    cutoff=args.cutoff,
                    edge_threshold=args.edge_threshold,
                    save_model=not args.no_save,
                )
                if result:
                    all_results.append(result)
            except FileNotFoundError as e:
                print(f"\n[SKIP] {asset}/{tf}: {e}")
            except Exception as e:
                print(f"\n[ERROR] {asset}/{tf}: {e}")
                raise

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Asset/TF':<12} {'Test Trades':>12} {'Win%':>8} {'Sharpe':>8}")
        print("-" * 44)
        for r in sorted(all_results, key=lambda x: x["test_sharpe"], reverse=True):
            label = f"{r['asset']}/{r['tf']}"
            print(f"{label:<12} {r['test_trades']:>12,} {r['test_win_rate']:>8.1%} {r['test_sharpe']:>8.2f}")

    # Save summary JSON
    if all_results and not args.no_save:
        summary_path = MODELS_DIR / "apex_ml_summary.json"
        with open(summary_path, "w") as f:
            # Exclude feature_coefs from summary to keep it compact
            slim = [{k: v for k, v in r.items() if k != "feature_coefs"} for r in all_results]
            json.dump(slim, f, indent=2)
        print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
