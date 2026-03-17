"""Run backtests for every (strategy, asset, timeframe) combination.

Usage:
    uv run python scripts/run_all_backtests.py
    uv run python scripts/run_all_backtests.py --enrich        # liq + funding only (~minutes)
    uv run python scripts/run_all_backtests.py --enrich --with-cvd  # also CVD (hours, 4yr data)

Data files must already exist as data/{asset}_{tf}.parquet.
Missing files are skipped with a warning. Results are written to
backtest_results/{strategy_name}_{asset}_{tf}/ and a combined
backtest_results/summary.json.

--enrich fetches liquidation and funding rate data from Binance (fast: sparse events).
--with-cvd additionally fetches aggTrades for CVD computation (very slow: millions of
  trades over a 4-year BTC dataset can take hours).
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd
from polymarket_algo.backtest.engine import parameter_sweep, run_backtest, walk_forward_split
from polymarket_algo.strategies import (
    CVDDivergenceStrategy,
    FundingRateExtremesStrategy,
    LiquidationCascadeStrategy,
    StreakADXStrategy,
    StreakReversalStrategy,
    StreakRSIStrategy,
)

# Strategies that work on plain OHLCV candles
STRATEGIES = [
    StreakReversalStrategy,
    StreakRSIStrategy,
    StreakADXStrategy,
]

# Strategies that require liq + funding enrichment only
LIQ_FUNDING_STRATEGIES = [
    LiquidationCascadeStrategy,
    FundingRateExtremesStrategy,
]

# Strategies that additionally require CVD (aggTrades — slow to fetch)
CVD_STRATEGIES = [
    CVDDivergenceStrategy,
]

TIMEFRAMES = ["5m", "15m", "1h"]
ASSETS = ["btc", "eth", "sol", "xrp"]
ENRICHED_ASSETS = ["btc", "eth"]  # assets with reliable Binance futures data

ASSET_TO_SYMBOL = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
}

STRATEGY_TARGETS = [(cls, asset, tf) for cls in STRATEGIES for asset in ASSETS for tf in TIMEFRAMES]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "backtest_results"


def load_candles(asset: str, timeframe: str) -> pd.DataFrame:
    path = DATA_DIR / f"{asset}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
    df = df.sort_index()
    return df


def run_strategy(
    strategy: object,
    candles: pd.DataFrame,
    asset: str,
    timeframe: str,
    summary: list[dict[str, Any]],
) -> None:
    """Run sweep + backtest for one (strategy, candles) combination and append to summary."""
    strategy_name = getattr(strategy, "name", type(strategy).__name__)
    print(f"[RUN ] {strategy_name} / {asset} / {timeframe} ...", end=" ", flush=True)

    if len(candles) < 50:
        print(f"too few candles ({len(candles)}), skipping")
        return

    train, test = walk_forward_split(candles, train_ratio=0.75)

    sweep_df = parameter_sweep(train, strategy, strategy.param_grid)  # type: ignore[arg-type]
    best_row = sweep_df.iloc[0].to_dict()
    best_params: dict[str, Any] = {k: best_row[k] for k in strategy.param_grid}  # type: ignore[union-attr]

    result = run_backtest(test, strategy, best_params)  # type: ignore[arg-type]

    sub_dir = OUT_DIR / f"{strategy_name}_{asset}_{timeframe}"
    sub_dir.mkdir(exist_ok=True)

    sweep_df.to_csv(sub_dir / "sweep.csv", index=False)
    result.trades.to_csv(sub_dir / "trades.csv", index=False)
    result.pnl_curve.rename("equity").to_csv(sub_dir / "equity.csv", index=True)

    row: dict[str, Any] = {
        "strategy": strategy_name,
        "asset": asset,
        "timeframe": timeframe,
        "best_params": best_params,
        **result.metrics,
    }
    summary.append(row)

    print(
        f"win_rate={result.metrics['win_rate']:.2%}  "
        f"pnl={result.metrics['total_pnl']:+.2f}  "
        f"sharpe={result.metrics['sharpe_ratio']:.2f}  "
        f"trades={result.metrics['trade_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all backtests")
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Also run liq/funding enriched strategies (LiquidationCascade, FundingRateExtremes). "
        "Fetches sparse event data from Binance — completes in minutes.",
    )
    parser.add_argument(
        "--with-cvd",
        action="store_true",
        dest="with_cvd",
        help="Also run CVDDivergenceStrategy. Requires fetching all aggTrades from Binance — "
        "expect several hours for a 4-year BTC dataset.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    summary: list[dict[str, Any]] = []

    # --- Plain OHLCV strategies ---
    for StrategyClass, asset, timeframe in STRATEGY_TARGETS:
        data_path = DATA_DIR / f"{asset}_{timeframe}.parquet"
        if not data_path.exists():
            print(f"[SKIP] Missing data: {data_path.name}")
            continue

        sig = inspect.signature(StrategyClass.__init__)
        if "asset" in sig.parameters:
            strategy = StrategyClass(asset=asset.upper())  # type: ignore[call-arg]
        else:
            strategy = StrategyClass()

        candles = load_candles(asset, timeframe)
        run_strategy(strategy, candles, asset, timeframe, summary)

    # --- Liq + Funding enriched strategies ---
    if args.enrich or args.with_cvd:
        from polymarket_algo.data import enrich_candles

        # Determine which strategy classes to run
        enriched_strategy_classes = list(LIQ_FUNDING_STRATEGIES)
        if args.with_cvd:
            enriched_strategy_classes.extend(CVD_STRATEGIES)

        # Build (asset, timeframe) pairs and cache enriched candles to avoid re-fetching
        asset_tf_pairs = [
            (asset, tf)
            for asset in ENRICHED_ASSETS
            for tf in TIMEFRAMES
            if (DATA_DIR / f"{asset}_{tf}.parquet").exists()
        ]

        enriched_cache: dict[tuple[str, str], pd.DataFrame] = {}

        for asset, timeframe in asset_tf_pairs:
            symbol = ASSET_TO_SYMBOL.get(asset, asset.upper() + "USDT")
            raw_candles = load_candles(asset, timeframe)
            print(
                f"[ENRICH] Fetching liq+funding for {symbol} {timeframe} ({len(raw_candles):,} candles)…",
                flush=True,
            )
            try:
                enriched = enrich_candles(
                    raw_candles,
                    symbol=symbol,
                    include_cvd=args.with_cvd,
                    include_liq=True,
                    include_funding=True,
                )
                enriched_cache[(asset, timeframe)] = enriched
            except Exception as exc:
                print(f"[ENRICH] Failed for {symbol} {timeframe}: {exc} — skipping")

        # Run each enriched strategy against cached enriched candles
        for StrategyClass in enriched_strategy_classes:
            strategy = StrategyClass()
            for asset, timeframe in asset_tf_pairs:
                candles = enriched_cache.get((asset, timeframe))
                if candles is None:
                    continue
                run_strategy(strategy, candles, asset, timeframe, summary)

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {len(summary)} results → {summary_path}")


if __name__ == "__main__":
    main()
