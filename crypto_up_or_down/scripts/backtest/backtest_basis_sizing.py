#!/usr/bin/env python3
"""Backtest basis strategy sizing options: flat vs z-linear vs z-band Kelly.

Compares three sizing approaches on a train/test walk-forward split:
  1. Flat         — fixed $BASE_SIZE regardless of signal strength (baseline)
  2. Z-linear     — size ∝ |basis_z| / z_thresh, capped at max_mult×
  3. Z-band Kelly — half-Kelly per z-band using train-measured win rates

Usage:
    uv run python scripts/backtest/backtest_basis_sizing.py
    uv run python scripts/backtest/backtest_basis_sizing.py --symbol ETHUSDT --days 365
    uv run python scripts/backtest/backtest_basis_sizing.py --z-thresh 1.5
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from polymarket_algo.backtest.engine import run_backtest, walk_forward_split
from polymarket_algo.data.binance import compute_basis_candles
from polymarket_algo.strategies.spot_perp_basis import SpotPerpBasisStrategy

# ── constants ──────────────────────────────────────────────────────────────────

LOOKBACK_DAYS = 730
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
Z_THRESH = 2.0
BASE_SIZE = 15.0

# Polymarket: win ~$0.45 on $0.50 risk → b ≈ 0.9; positive Kelly needs p > 52.6%
_B = 0.9
_BREAKEVEN_P = 1.0 / (1.0 + _B)  # ≈ 0.526

# Normalise Kelly sizes relative to the streak-reversal reference point.
# BTC 5m trigger=4 half-Kelly fraction (from packages/strategies/_ci_sizing.py).
_F_REF = 0.014

# Z-score bands for empirical win-rate measurement
Z_BANDS: list[tuple[float, float]] = [
    (2.0, 2.5),
    (2.5, 3.0),
    (3.0, 3.5),
    (3.5, float("inf")),
]

# Max size multipliers to sweep for z-linear approach
Z_LINEAR_MAX_MULTS = [1.5, 2.0, 2.5, 3.0, 4.0]

_VISION_SPOT_URL = "https://data-api.binance.vision/api/v3/klines"
_FAPI_PERP_URL = "https://fapi.binance.com/fapi/v1/klines"

# ── data fetching ──────────────────────────────────────────────────────────────


def _fetch_spot_vision(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch spot OHLCV via Binance Vision (no geo-block, no credentials)."""
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            _VISION_SPOT_URL,
            params={"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        last_open = int(data[-1][0])
        if last_open <= cursor:
            break
        cursor = last_open + 1
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def _fetch_perp_fapi(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch perp OHLCV from Binance futures API (fapi.binance.com)."""
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            _FAPI_PERP_URL,
            params={"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        last_open = int(data[-1][0])
        if last_open <= cursor:
            break
        cursor = last_open + 1
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def load_basis_candles(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Load or fetch spot+perp candles and return enriched DataFrame with basis_zscore."""
    spot_path = Path("data") / f"{symbol.lower()}_{interval}_spot.parquet"
    perp_path = Path("data") / f"{symbol.lower()}_{interval}_perp.parquet"

    now = datetime.now(tz=UTC)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    Path("data").mkdir(exist_ok=True)

    # --- spot ---
    if spot_path.exists():
        print(f"  Loading cached spot from {spot_path}...")
        spot_raw = pd.read_parquet(spot_path)
        spot_raw["open_time"] = pd.to_datetime(spot_raw["open_time"], utc=True)
        cutoff = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        spot = spot_raw[spot_raw["open_time"] >= cutoff].set_index("open_time").sort_index()
    else:
        print(f"  Fetching {symbol} {interval} spot ({days}d)...")
        spot_df = _fetch_spot_vision(symbol, interval, start_ms, end_ms)
        spot_df.to_parquet(spot_path)
        spot = spot_df.set_index("open_time").sort_index()

    # --- perp ---
    if perp_path.exists():
        print(f"  Loading cached perp from {perp_path}...")
        perp_raw = pd.read_parquet(perp_path)
        perp_raw["open_time"] = pd.to_datetime(perp_raw["open_time"], utc=True)
        cutoff = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        perp_raw = perp_raw[perp_raw["open_time"] >= cutoff]
        perp = perp_raw  # keep as DataFrame with open_time column for compute_basis_candles
    else:
        print(f"  Fetching {symbol} {interval} perp ({days}d)...")
        perp = _fetch_perp_fapi(symbol, interval, start_ms, end_ms)
        perp.to_parquet(perp_path)

    print(f"  Spot: {len(spot):,} candles  Perp: {len(perp):,} candles")

    enriched = compute_basis_candles(perp, spot)
    print(f"  Enriched: {len(enriched):,} rows  "
          f"({enriched.index[0].date()} → {enriched.index[-1].date()})")
    return enriched


# ── strategy variants ──────────────────────────────────────────────────────────


class BasisZLinearStrategy:
    """Sizes each trade proportionally to |basis_z| / z_thresh, capped at max_mult."""

    name = "basis_z_linear"
    _base = SpotPerpBasisStrategy()

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        z_thresh = float(params.get("z_thresh", Z_THRESH))
        base_size = float(params.get("size", BASE_SIZE))
        max_mult = float(params.get("max_mult", 3.0))

        result = self._base.evaluate(candles, z_thresh=z_thresh, size=base_size).copy()

        if "basis_zscore" not in candles.columns:
            return result

        z_abs = candles["basis_zscore"].abs()
        # Linear: 1× at threshold, grows proportionally, capped
        multiplier = (z_abs / z_thresh).clip(upper=max_mult)
        active = result["signal"] != 0
        result.loc[active, "size"] = base_size * multiplier[active]
        return result


class BasisZBandKellyStrategy:
    """Sizes each trade using half-Kelly based on train-measured win rates per z-band.

    Trades in bands with insufficient edge (p < breakeven) are skipped (size=0).
    """

    name = "basis_z_band_kelly"
    _base = SpotPerpBasisStrategy()

    def __init__(self, band_rates: dict[tuple[float, float], float | None], max_size: float = 3 * BASE_SIZE):
        # band_rates: {(lo, hi): win_rate} — None means insufficient data
        self.band_rates = band_rates
        self.max_size = max_size

    def _lookup_rate(self, abs_z: float) -> float | None:
        for (lo, hi), rate in self.band_rates.items():
            if lo <= abs_z < hi:
                return rate
        return None

    def _half_kelly_size(self, p: float) -> float:
        if p <= _BREAKEVEN_P:
            return 0.0
        f_half = 0.5 * max((p * _B - (1 - p)) / _B, 0.0)
        if _F_REF <= 0 or f_half <= 0:
            return 0.0
        return min(BASE_SIZE * f_half / _F_REF, self.max_size)

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        z_thresh = float(params.get("z_thresh", Z_THRESH))
        base_size = float(params.get("size", BASE_SIZE))

        result = self._base.evaluate(candles, z_thresh=z_thresh, size=base_size).copy()

        if "basis_zscore" not in candles.columns or not self.band_rates:
            return result

        z_abs = candles["basis_zscore"].abs()
        active_idx = result.index[result["signal"] != 0]

        for idx in active_idx:
            p = self._lookup_rate(float(z_abs.loc[idx]))
            if p is None:
                # No data for this band — use flat fallback
                result.loc[idx, "size"] = base_size
            else:
                result.loc[idx, "size"] = self._half_kelly_size(p)

        return result


# ── analysis helpers ───────────────────────────────────────────────────────────


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return centre - margin, centre + margin


def compute_band_rates(
    trades: pd.DataFrame,
    candles: pd.DataFrame,
) -> dict[tuple[float, float], tuple[float, float, float, int] | None]:
    """Return per-band (win_rate, ci_lo, ci_hi, n) from a trades DataFrame."""
    if trades.empty or "basis_zscore" not in candles.columns:
        return {band: None for band in Z_BANDS}

    z_abs = candles["basis_zscore"].abs()
    # trades.timestamp is the index value from candles
    trades_z = trades.copy()
    trades_z["abs_z"] = trades_z["timestamp"].map(z_abs)

    result: dict[tuple[float, float], tuple[float, float, float, int] | None] = {}
    for lo, hi in Z_BANDS:
        mask = (trades_z["abs_z"] >= lo) & (trades_z["abs_z"] < hi)
        band = trades_z[mask]
        n = len(band)
        if n >= 30:
            wins = int(band["is_win"].sum())
            rate = wins / n
            ci_lo, ci_hi = wilson_ci(wins, n)
            result[(lo, hi)] = (rate, ci_lo, ci_hi, n)
        else:
            result[(lo, hi)] = None
    return result


def print_band_table(
    band_data: dict[tuple[float, float], tuple[float, float, float, int] | None],
    symbol: str,
    interval: str,
) -> None:
    asset = symbol.replace("USDT", "")
    print(f"\n{'=' * 75}")
    print(f"Z-BAND WIN RATES — train set  ({asset} {interval}, flat strategy)")
    print(f"  Breakeven at p={_BREAKEVEN_P:.1%} (Polymarket b={_B})")
    print(f"{'=' * 75}")
    hdr = f"  {'Band':>12}  {'Trades':>7}  {'Win Rate':>9}  {'95% CI':>17}  {'Half-Kelly f':>13}  {'Size mult':>10}"
    print(hdr)
    print(f"  {'-' * 12}  {'-' * 7}  {'-' * 9}  {'-' * 17}  {'-' * 13}  {'-' * 10}")

    for (lo, hi), data in band_data.items():
        hi_str = f"{hi:.1f}" if hi != float("inf") else "∞"
        band_label = f"[{lo:.1f}, {hi_str})"
        if data is None:
            print(f"  {band_label:>12}  {'<30':>7}  {'—':>9}  {'—':>17}  {'—':>13}  {'—':>10}")
            continue
        rate, ci_lo, ci_hi, n = data
        ci_str = f"[{ci_lo:.1%}, {ci_hi:.1%}]"

        f_half = 0.5 * max((rate * _B - (1 - rate)) / _B, 0.0) if rate > _BREAKEVEN_P else 0.0
        size_mult = f_half / _F_REF if f_half > 0 else 0.0

        f_str = f"{f_half:.4f}" if f_half > 0 else "no edge"
        mult_str = f"{size_mult:.2f}×" if size_mult > 0 else "skip"
        print(f"  {band_label:>12}  {n:>7,}  {rate:>9.1%}  {ci_str:>17}  {f_str:>13}  {mult_str:>10}")
    print()


def print_result(label: str, metrics: dict, width: int = 20) -> None:
    m = metrics
    print(f"  {label:<{width}}  trades={m['trade_count']:>5}  win={m['win_rate']:.1%}"
          f"  pnl=${m['total_pnl']:>+9.2f}  sharpe={m['sharpe_ratio']:>6.3f}"
          f"  maxdd=${m['max_drawdown']:>8.2f}")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest basis sizing options")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=INTERVAL, choices=["5m", "15m"])
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--z-thresh", type=float, default=Z_THRESH, dest="z_thresh")
    parser.add_argument("--base-size", type=float, default=BASE_SIZE, dest="base_size")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached parquet files")
    args = parser.parse_args()

    z_thresh = args.z_thresh
    base_size = args.base_size
    asset = args.symbol.replace("USDT", "")

    if args.no_cache:
        for p in Path("data").glob(f"{args.symbol.lower()}_{args.interval}_*.parquet"):
            p.unlink()

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"\nLoading {args.symbol} {args.interval} basis candles ({args.days}d)...")
    candles = load_basis_candles(args.symbol, args.interval, args.days)
    train, test = walk_forward_split(candles)
    print(f"  Train: {len(train):,} candles | Test: {len(test):,} candles\n")

    flat_strategy = SpotPerpBasisStrategy()
    eval_params = {"z_thresh": z_thresh, "size": base_size}

    # ── PHASE 1: train-set analysis ───────────────────────────────────────────
    print(f"{'=' * 75}")
    print(f"PHASE 1 — TRAIN SET  ({asset} {args.interval}, z_thresh={z_thresh}, base=${base_size:.0f})")
    print(f"{'=' * 75}")

    train_flat = run_backtest(train, flat_strategy, eval_params)
    print("Flat strategy (train):")
    print_result("flat", train_flat.metrics)

    # Z-band win rates from train trades
    band_data = compute_band_rates(train_flat.trades, train)
    print_band_table(band_data, args.symbol, args.interval)

    # Z-linear sweep on train
    print(f"{'=' * 75}")
    print(f"Z-LINEAR max_mult SWEEP — train set")
    print(f"{'=' * 75}")
    z_linear_strategy = BasisZLinearStrategy()
    best_mult = Z_LINEAR_MAX_MULTS[0]
    best_train_sharpe = -float("inf")
    for max_mult in Z_LINEAR_MAX_MULTS:
        r = run_backtest(train, z_linear_strategy, {**eval_params, "max_mult": max_mult})
        label = f"z_linear max_mult={max_mult:.1f}×"
        print_result(label, r.metrics)
        if r.metrics["sharpe_ratio"] > best_train_sharpe:
            best_train_sharpe = r.metrics["sharpe_ratio"]
            best_mult = max_mult
    print(f"\n  → Best max_mult on train: {best_mult}× (Sharpe {best_train_sharpe:.3f})\n")

    # Build z-band Kelly strategy from train rates
    band_win_rates: dict[tuple[float, float], float | None] = {}
    for band, data in band_data.items():
        band_win_rates[band] = data[0] if data is not None else None

    zband_strategy = BasisZBandKellyStrategy(band_win_rates, max_size=3 * base_size)

    train_zband = run_backtest(train, zband_strategy, eval_params)
    print(f"{'=' * 75}")
    print("Z-BAND KELLY — train set")
    print(f"{'=' * 75}")
    print_result("z_band_kelly", train_zband.metrics)
    print()

    # ── PHASE 2: test-set comparison ──────────────────────────────────────────
    print(f"{'=' * 75}")
    print(f"PHASE 2 — TEST SET COMPARISON  (held-out 25%)")
    print(f"{'=' * 75}")

    # baseline
    test_flat = run_backtest(test, flat_strategy, eval_params)

    # z-linear (best max_mult from train)
    test_zlinear = run_backtest(test, z_linear_strategy, {**eval_params, "max_mult": best_mult})

    # z-band Kelly (rates from train)
    test_zband = run_backtest(test, zband_strategy, eval_params)

    print_result("flat (baseline)", test_flat.metrics)
    print_result(f"z_linear ({best_mult:.1f}×)", test_zlinear.metrics)
    print_result("z_band_kelly", test_zband.metrics)

    # ── PHASE 3: per-band test breakdown ─────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("Z-BAND WIN RATES — test set  (out-of-sample verification)")
    print(f"{'=' * 75}")
    test_band_data = compute_band_rates(test_flat.trades, test)
    print_band_table(test_band_data, args.symbol, args.interval)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"{'=' * 75}")
    print("SUMMARY")
    print(f"{'=' * 75}")

    results = [
        ("flat (baseline)", test_flat.metrics),
        (f"z_linear ({best_mult:.1f}×)", test_zlinear.metrics),
        ("z_band_kelly", test_zband.metrics),
    ]
    best_label, best_metrics = max(results, key=lambda x: x[1]["sharpe_ratio"])
    print(f"\n  Best by Sharpe on test set: {best_label}")
    print(f"    Sharpe  : {best_metrics['sharpe_ratio']:.3f}")
    print(f"    Total PnL: ${best_metrics['total_pnl']:+.2f}")
    print(f"    Win rate : {best_metrics['win_rate']:.1%}")
    print(f"    Trades   : {best_metrics['trade_count']}")
    print(f"    Max DD   : ${best_metrics['max_drawdown']:.2f}")

    # Check if z-band Kelly has sufficient data in all bands
    any_none = any(v is None for v in band_win_rates.values())
    if any_none:
        print(f"\n  NOTE: some z-bands had <30 train trades — z-band Kelly used flat fallback there.")
        print(f"        Consider running with --days {args.days * 2} for more data.")
    print()


if __name__ == "__main__":
    main()
