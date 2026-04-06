"""Merge trade records onto OHLCV candles by timestamp."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from deepanal.models import TradeRecord

ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data"

# Binance symbol map: asset+kind → symbol
_SYMBOL_MAP = {
    ("btc", "perp"): "BTCUSDT",
    ("btc", "spot"): "BTCUSDT",
    ("eth", "perp"): "ETHUSDT",
    ("eth", "spot"): "ETHUSDT",
}


def _refresh_ohlcv(path: Path, asset: str, tf: str) -> None:
    """Append any candles newer than the last row in *path* from the exchange."""
    try:
        from polymarket_algo.data.binance import fetch_klines

        existing = pd.read_parquet(path)
        existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
        last_ts = existing["open_time"].max()

        start_ms = int(last_ts.timestamp() * 1000) + 1
        end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)

        symbol = _SYMBOL_MAP.get((asset.lower(), "perp"), f"{asset.upper()}USDT")
        new_df = fetch_klines(symbol, tf, start_ms, end_ms)

        if new_df.empty:
            return

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True)
        combined = (
            combined.drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        combined.to_parquet(path, index=False)
        print(f"[aligner] fetched {len(new_df):,} new {tf} candles for {asset}", flush=True)
    except Exception as exc:
        print(f"[aligner] refresh skipped: {exc}", flush=True)


def load_ohlcv(asset: str = "btc", tf: str = "5m", kind: str = "perp") -> pd.DataFrame:
    """Load OHLCV from parquet, auto-refreshing stale data from the exchange.

    Args:
        asset: e.g. "btc"
        tf:    candle timeframe, e.g. "5m"
        kind:  "perp" or "spot"
    """
    path = DATA_DIR / f"{asset}usdt_{tf}_{kind}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"OHLCV not found: {path}")

    _refresh_ohlcv(path, asset, tf)

    df = pd.read_parquet(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    """Convert list[TradeRecord] to a flat DataFrame sorted by open_time."""
    if not trades:
        return pd.DataFrame()
    rows = [t.__dict__ for t in trades]
    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.sort_values("open_time").reset_index(drop=True)


def align(
    trades: list[TradeRecord],
    ohlcv: pd.DataFrame,
    padding: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge trades onto OHLCV candles.

    For each trade row, attaches the candle that opened at or just before the
    trade's open_time using merge_asof (backward).

    Returns:
        ohlcv_window : OHLCV slice ±padding candles around the trade date range
        trades_df    : trades DataFrame enriched with c_open/c_high/c_low/c_close/c_volume
    """
    trades_df = to_dataframe(trades)
    if trades_df.empty:
        return ohlcv.copy(), trades_df

    t_min = trades_df["open_time"].min()
    t_max = trades_df["open_time"].max()

    idx = ohlcv.index
    lo = max(0, idx.searchsorted(t_min) - padding)
    hi = min(len(idx), idx.searchsorted(t_max) + padding + 1)
    ohlcv_window = ohlcv.iloc[lo:hi].copy()

    # Build candle lookup with renamed columns to avoid collision
    candle_cols = [c for c in ("open", "high", "low", "close", "volume") if c in ohlcv_window.columns]
    candle_snap = ohlcv_window[candle_cols].copy()
    candle_snap.columns = [f"c_{c}" for c in candle_cols]
    candle_snap = candle_snap.reset_index()  # open_time becomes a column

    # Normalise both sides to the same datetime resolution before merge
    trades_df["open_time"] = trades_df["open_time"].dt.as_unit("ns")
    candle_snap["open_time"] = candle_snap["open_time"].dt.as_unit("ns")

    trades_df = pd.merge_asof(
        trades_df.sort_values("open_time"),
        candle_snap,
        on="open_time",
        direction="backward",
    )
    return ohlcv_window, trades_df
