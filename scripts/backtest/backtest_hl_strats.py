#!/usr/bin/env python3
"""Backtest all HL orderflow strategies on historical Binance candle data.

Fetches 90 days of candles, pre-enriches ONCE with enrich_with_hl_orderflow()
(~4 HTTP calls to HL), then runs walk-forward backtests for:
  - HLOrderFlowMomentumStrategy
  - HLOrderFlowReversalStrategy
  - DeltaFlipStrategy
  - ThreeBarMoMoStrategy (hl_gate=True)
  - PinBarReversalStrategy (hl_gate=True)

Results are contamination-free: each row uses the HL snapshot that existed at
that candle's open_time, not today's live data.
"""

import argparse
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests
from polymarket_algo.backtest.engine import parameter_sweep, run_backtest, walk_forward_split
from polymarket_algo.data.hl_enrich import enrich_with_hl_orderflow
from polymarket_algo.strategies.delta_flip import DeltaFlipStrategy
from polymarket_algo.strategies.hl_orderflow_momentum import HLOrderFlowMomentumStrategy
from polymarket_algo.strategies.hl_orderflow_reversal import HLOrderFlowReversalStrategy
from polymarket_algo.strategies.pin_bar import PinBarReversalStrategy
from polymarket_algo.strategies.three_bar_momo import ThreeBarMoMoStrategy

LOOKBACK_DAYS = 90
_VISION_URL = "https://data-api.binance.vision/api/v3/klines"


def _fetch_klines_vision(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            _VISION_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        last_open = data[-1][0]
        if last_open <= cursor:
            break
        cursor = last_open + 1
        time.sleep(0.05)

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def _print_metrics(label: str, m: dict) -> None:
    print(f"  Trade count : {m['trade_count']}")
    print(f"  Win rate    : {m['win_rate']:.1%}")
    print(f"  Total PnL   : ${m['total_pnl']:+.2f}")
    print(f"  Max drawdown: ${m['max_drawdown']:.2f}")
    print(f"  Sharpe      : {m['sharpe_ratio']:.3f}")


def _section(title: str) -> None:
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def backtest_strategy(name, strategy, enriched, train, test, param_grid, fixed_params=None):
    _section(name)
    fixed = fixed_params or {}

    # Full dataset with default params
    default_result = run_backtest(enriched, strategy, fixed)
    print(f"Default params ({fixed or 'strategy defaults'}) — full {len(enriched):,} candles:")
    _print_metrics(name, default_result.metrics)

    if not param_grid:
        return

    # Sweep on train
    print(f"\nParam sweep on train ({len(train):,} candles)...")
    sweep_params = {**param_grid}
    # Inject fixed params into every sweep combo
    if fixed:
        full_sweep = []
        from itertools import product as iproduct
        keys = list(sweep_params.keys())
        values = list(sweep_params.values())
        combos = [{**dict(zip(keys, combo)), **fixed} for combo in iproduct(*values)]
        # Build a flat grid for parameter_sweep
        for k, v in fixed.items():
            if k not in sweep_params:
                sweep_params[k] = [v]

    sweep = parameter_sweep(train, strategy, sweep_params)
    top5 = sweep.head(5)
    cols = [c for c in ["win_rate", "total_pnl", "trade_count", "sharpe_ratio"] + list(sweep_params.keys()) if c in sweep.columns]
    print(f"Top 5 by Sharpe:")
    print(top5[cols].to_string(index=False))

    # Best on test
    best_row = sweep.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in sweep_params if k in best_row}
    # Cast ints
    for k in ["min_votes", "bars"]:
        if k in best_params:
            best_params[k] = int(best_params[k])

    print(f"\nBest params {best_params} on test ({len(test):,} candles):")
    test_result = run_backtest(test, strategy, best_params)
    _print_metrics(name, test_result.metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest HL orderflow strategies")
    parser.add_argument("--coin", default="BTC", choices=["BTC", "ETH", "SOL", "XRP"])
    parser.add_argument("--interval", default="5m", choices=["5m", "15m", "1h"])
    args = parser.parse_args()

    SYMBOL = f"{args.coin}USDT"
    INTERVAL = args.interval
    COIN = args.coin

    # --- 1. Fetch candles ---
    now = datetime.now(tz=UTC)
    start = now - timedelta(days=LOOKBACK_DAYS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    print(f"Fetching {SYMBOL} {INTERVAL} — {LOOKBACK_DAYS} days...")
    raw = _fetch_klines_vision(SYMBOL, INTERVAL, start_ms, end_ms)
    candles = raw.set_index("open_time").sort_index()
    print(f"  {len(candles):,} candles  ({candles.index[0].date()} → {candles.index[-1].date()})")

    # --- 2. Pre-enrich with HL orderflow (single fetch for all strategies) ---
    print(f"\nFetching HL orderflow history for {COIN} across 4 timeframes...")
    t0 = time.time()
    enriched = enrich_with_hl_orderflow(candles, coin=COIN)
    hl_cols = [c for c in enriched.columns if c.startswith("hl_")]
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(hl_cols)} HL columns added")
    print(f"  Columns: {hl_cols[:4]}...")

    # Verify per-row variation (contamination check)
    side_col = f"hl_{COIN}_5m_dominant_side"
    vc = enriched[side_col].value_counts()
    print(f"\n  HL 5m dominant_side distribution (contamination check):")
    for side, count in vc.items():
        print(f"    {side}: {count:,} rows ({count/len(enriched):.1%})")
    if enriched[side_col].nunique() == 1:
        print("  WARNING: all rows have same dominant_side — possible contamination!")
    else:
        print("  OK: per-row variation confirmed (contamination fixed)")

    # --- 3. Train/test split ---
    train, test = walk_forward_split(enriched)
    print(f"\nTrain: {len(train):,} candles | Test: {len(test):,} candles")

    # --- 4. HLOrderFlowMomentumStrategy ---
    backtest_strategy(
        f"HLOrderFlowMomentumStrategy ({COIN}/{INTERVAL})",
        HLOrderFlowMomentumStrategy(),
        enriched, train, test,
        param_grid={"min_votes": [2, 3], "size": [10.0, 15.0, 20.0]},
        fixed_params={"coin": COIN},
    )

    # --- 5. HLOrderFlowReversalStrategy ---
    backtest_strategy(
        f"HLOrderFlowReversalStrategy ({COIN}/{INTERVAL})",
        HLOrderFlowReversalStrategy(),
        enriched, train, test,
        param_grid={"min_votes": [2, 3], "size": [10.0, 15.0, 20.0]},
        fixed_params={"coin": COIN},
    )

    # --- 6. DeltaFlipStrategy ---
    backtest_strategy(
        f"DeltaFlipStrategy ({COIN}/{INTERVAL}, gate=4h)",
        DeltaFlipStrategy(),
        enriched, train, test,
        param_grid={"base_size": [1.0, 2.0, 5.0, 10.0, 15.0]},
        fixed_params={"coin": COIN, "timeframe": INTERVAL, "gate_timeframe": "4h"},
    )

    # --- 7. ThreeBarMoMo + HL gate vs without ---
    _section("ThreeBarMoMoStrategy — no gate vs HL gate")

    no_gate = run_backtest(enriched, ThreeBarMoMoStrategy(), {"bars": 3, "size": 15.0, "hl_gate": False})
    hl_gate = run_backtest(enriched, ThreeBarMoMoStrategy(), {"bars": 3, "size": 15.0, "hl_gate": True, "hl_coin": COIN})
    print("No HL gate — full dataset:")
    _print_metrics("", no_gate.metrics)
    print("\nWith HL gate — full dataset:")
    _print_metrics("", hl_gate.metrics)

    sweep_no_gate = parameter_sweep(train, ThreeBarMoMoStrategy(), {"bars": [2,3,4], "size": [10.0,15.0,20.0], "hl_gate": [False]})
    sweep_hl_gate = parameter_sweep(train, ThreeBarMoMoStrategy(), {"bars": [2,3,4], "size": [10.0,15.0,20.0], "hl_gate": [True], "hl_coin": [COIN]})

    best_no_gate = {k: sweep_no_gate.iloc[0][k] for k in ["bars","size","hl_gate"]}
    best_no_gate["bars"] = int(best_no_gate["bars"])
    best_hl_gate = {k: sweep_hl_gate.iloc[0][k] for k in ["bars","size","hl_gate","hl_coin"]}
    best_hl_gate["bars"] = int(best_hl_gate["bars"])

    print(f"\nBest no-gate {best_no_gate} on test:")
    _print_metrics("", run_backtest(test, ThreeBarMoMoStrategy(), best_no_gate).metrics)
    print(f"\nBest HL-gate {best_hl_gate} on test:")
    _print_metrics("", run_backtest(test, ThreeBarMoMoStrategy(), best_hl_gate).metrics)

    # --- 8. PinBarReversal + HL gate vs without ---
    _section("PinBarReversalStrategy — no gate vs HL gate")

    no_gate = run_backtest(enriched, PinBarReversalStrategy(), {"size": 15.0, "hl_gate": False})
    hl_gate = run_backtest(enriched, PinBarReversalStrategy(), {"size": 15.0, "hl_gate": True, "hl_coin": COIN})
    print("No HL gate — full dataset:")
    _print_metrics("", no_gate.metrics)
    print("\nWith HL gate — full dataset:")
    _print_metrics("", hl_gate.metrics)

    sweep_no_gate = parameter_sweep(train, PinBarReversalStrategy(), {
        "body_threshold": [0.25, 0.35, 0.45], "wick_threshold": [0.50, 0.55, 0.60],
        "size": [15.0], "hl_gate": [False]
    })
    sweep_hl_gate = parameter_sweep(train, PinBarReversalStrategy(), {
        "body_threshold": [0.25, 0.35, 0.45], "wick_threshold": [0.50, 0.55, 0.60],
        "size": [15.0], "hl_gate": [True], "hl_coin": [COIN]
    })

    best_no_gate = {k: sweep_no_gate.iloc[0][k] for k in ["body_threshold","wick_threshold","size","hl_gate"]}
    best_hl_gate = {k: sweep_hl_gate.iloc[0][k] for k in ["body_threshold","wick_threshold","size","hl_gate","hl_coin"]}

    print(f"\nBest no-gate {best_no_gate} on test:")
    _print_metrics("", run_backtest(test, PinBarReversalStrategy(), best_no_gate).metrics)
    print(f"\nBest HL-gate {best_hl_gate} on test:")
    _print_metrics("", run_backtest(test, PinBarReversalStrategy(), best_hl_gate).metrics)

    print()
    print("=" * 62)
    print("Done.")


if __name__ == "__main__":
    main()
