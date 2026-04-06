"""trade_context.py — Join trade records with surrounding OHLCV candles.

Usage examples
--------------
# Dump all trades with 20-candle pre-entry context to CSV
uv run python scripts/analysis/trade_context.py --out /tmp/trades_ctx.csv

# Filter by strategy, print to stdout
uv run python scripts/analysis/trade_context.py --strategy streak --print

# Filter by outcome, custom context window
uv run python scripts/analysis/trade_context.py --won --candles-before 30 --candles-after 10

# Specific bot
uv run python scripts/analysis/trade_context.py --bot 3barmomo-5m-scale --out /tmp/3bar.csv

# Aggregate stats by strategy (win-rate, pnl, entry price context)
uv run python scripts/analysis/trade_context.py --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]
STATE_DIR = Path("/opt/polymarket/state")
if not STATE_DIR.exists():
    STATE_DIR = ROOT / "state"

DATA_DIR = ROOT / "data"

# OHLCV parquet files keyed by (symbol, interval)
OHLCV_FILES: dict[tuple[str, str], Path] = {
    ("btc", "5m"):  DATA_DIR / "btc_5m.parquet",
    ("btc", "15m"): DATA_DIR / "btc_15m.parquet",
    ("btc", "1h"):  DATA_DIR / "btc_1h.parquet",
    ("btc", "4h"):  DATA_DIR / "btc_4h.parquet",
    ("eth", "5m"):  DATA_DIR / "eth_5m.parquet",
    ("eth", "15m"): DATA_DIR / "eth_15m.parquet",
    ("eth", "1h"):  DATA_DIR / "eth_1h.parquet",
    ("eth", "4h"):  DATA_DIR / "eth_4h.parquet",
    ("btc", "5m-perp"): DATA_DIR / "btcusdt_5m_perp.parquet",
    ("btc", "5m-spot"): DATA_DIR / "btcusdt_5m_spot.parquet",
}

# Default OHLCV to use when joining trades (BTC 5m covers most strategies)
DEFAULT_SYMBOL = "btc"
DEFAULT_INTERVAL = "5m"


# ── loaders ───────────────────────────────────────────────────────────────────

def load_trades(bot: str | None = None) -> pd.DataFrame:
    """Load all trades from state dir into a flat DataFrame."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from deepanal.loader import discover_bots, load_bot_trades  # type: ignore

    bots = discover_bots(STATE_DIR)
    if bot:
        bots = [b for b in bots if b == bot or bot in b]
        if not bots:
            print(f"No bot matching '{bot}'. Available: {discover_bots(STATE_DIR)}")
            sys.exit(1)

    records = []
    for b in bots:
        for t in load_bot_trades(b, STATE_DIR):
            records.append({
                "id":                  t.id,
                "bot":                 b,
                "strategy":            t.strategy,
                "source":              t.source,
                "open_time":           t.open_time,
                "executed_at":         t.executed_at,
                "direction":           t.direction,
                "amount":              t.amount,
                "entry_price":         t.entry_price,
                "fill_price":          t.fill_price,
                "won":                 t.won,
                "pnl":                 t.pnl,
                "hour_utc":            t.hour_utc,
                "day_of_week":         t.day_of_week,
                "consecutive_wins":    t.consecutive_wins,
                "consecutive_losses":  t.consecutive_losses,
                "bankroll_before":     t.bankroll_before,
                "gate_name":           t.gate_name,
                "gate_boosted":        t.gate_boosted,
                "slippage_pct":        t.slippage_pct,
                "fill_pct":            t.fill_pct,
                "market_bias":         t.market_bias,
                "is_paper":            t.is_paper,
                "market_slug":         t.market_slug,
            })

    if not records:
        print("No trades found.")
        sys.exit(0)

    df = pd.DataFrame(records)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates(subset=["id"]).sort_values("open_time").reset_index(drop=True)
    return df


def load_ohlcv(symbol: str = DEFAULT_SYMBOL, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    key = (symbol.lower(), interval.lower())
    path = OHLCV_FILES.get(key)
    if path is None or not path.exists():
        available = [f"{s}/{i}" for (s, i) in OHLCV_FILES if OHLCV_FILES[(s, i)].exists()]
        print(f"No OHLCV file for {symbol}/{interval}. Available: {available}")
        sys.exit(1)
    ohlcv = pd.read_parquet(path, columns=["open_time", "open", "high", "low", "close", "volume"])
    ohlcv["open_time"] = pd.to_datetime(ohlcv["open_time"], utc=True)
    return ohlcv.sort_values("open_time").reset_index(drop=True)


# ── context join ──────────────────────────────────────────────────────────────

def build_context(
    trades: pd.DataFrame,
    ohlcv: pd.DataFrame,
    candles_before: int = 20,
    candles_after: int = 5,
) -> pd.DataFrame:
    """For each trade, attach pre/post OHLCV candle stats as flat columns.

    Adds columns:
      ctx_open_N, ctx_high_N, ctx_low_N, ctx_close_N, ctx_vol_N
        where N is -candles_before .. candles_after (negative = before entry)

      pre_ret       — return from ctx_close[-candles_before] to entry candle close
      pre_vol_mean  — mean volume over the pre-entry window
      pre_vol_last  — volume of the candle just before entry
      vol_spike     — pre_vol_last / pre_vol_mean  (>2 = spike)
      pre_range     — (max_high - min_low) / entry_close over pre window
      entry_candle_ret — return of the entry candle itself (close/open - 1)
    """
    ohlcv_idx = ohlcv.set_index("open_time")
    times = ohlcv_idx.index  # sorted DatetimeIndex

    rows = []
    for _, trade in trades.iterrows():
        t_time = trade["open_time"]

        # Find position of this candle in the OHLCV index
        pos = times.searchsorted(t_time, side="left")
        if pos >= len(times) or times[pos] != t_time:
            # Snap to nearest candle within 5 min tolerance
            pos = times.searchsorted(t_time, side="nearest") if hasattr(times, "searchsorted") else pos

        pre_start = max(0, pos - candles_before)
        post_end  = min(len(ohlcv_idx), pos + candles_after + 1)

        pre_slice  = ohlcv_idx.iloc[pre_start:pos]
        post_slice = ohlcv_idx.iloc[pos:post_end]
        entry_candle = ohlcv_idx.iloc[pos] if pos < len(ohlcv_idx) else None

        ctx: dict = dict(trade)

        # Flat candle columns: c_-20_close, c_-1_close, c_0_close, c_1_close …
        for offset, slc in [(-candles_before, pre_slice), (1, post_slice)]:
            for i, (ts, row) in enumerate(slc.iterrows()):
                n = offset + i
                ctx[f"c_{n:+d}_open"]  = row["open"]
                ctx[f"c_{n:+d}_high"]  = row["high"]
                ctx[f"c_{n:+d}_low"]   = row["low"]
                ctx[f"c_{n:+d}_close"] = row["close"]
                ctx[f"c_{n:+d}_vol"]   = row["volume"]

        # Summary stats
        if len(pre_slice) > 0:
            entry_close = entry_candle["close"] if entry_candle is not None else float("nan")
            ctx["pre_ret"]       = (entry_close / pre_slice.iloc[0]["close"] - 1) if pre_slice.iloc[0]["close"] else float("nan")
            ctx["pre_vol_mean"]  = pre_slice["volume"].mean()
            ctx["pre_vol_last"]  = pre_slice.iloc[-1]["volume"]
            ctx["vol_spike"]     = ctx["pre_vol_last"] / ctx["pre_vol_mean"] if ctx["pre_vol_mean"] else float("nan")
            ctx["pre_range"]     = (pre_slice["high"].max() - pre_slice["low"].min()) / entry_close if entry_close else float("nan")
            if entry_candle is not None:
                ctx["entry_candle_ret"] = (entry_candle["close"] / entry_candle["open"] - 1) if entry_candle["open"] else float("nan")

        rows.append(ctx)

    return pd.DataFrame(rows)


# ── stats ─────────────────────────────────────────────────────────────────────

def print_stats(df: pd.DataFrame) -> None:
    """Print aggregate stats grouped by strategy."""
    settled = df[df["won"].notna()].copy()
    if settled.empty:
        print("No settled trades.")
        return

    settled["won"] = settled["won"].astype(bool)

    grp = settled.groupby("strategy").agg(
        trades=("id", "count"),
        win_rate=("won", "mean"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        avg_entry_price=("entry_price", "mean"),
        avg_vol_spike=("vol_spike", "mean") if "vol_spike" in settled.columns else ("pnl", "mean"),
    ).round(4)

    grp["win_rate"] = (grp["win_rate"] * 100).round(1).astype(str) + "%"
    grp["total_pnl"] = grp["total_pnl"].map("${:.2f}".format)
    grp["avg_pnl"]   = grp["avg_pnl"].map("${:.2f}".format)

    print(grp.to_string())
    print(f"\nTotal settled trades: {len(settled)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Join trades with OHLCV context")
    parser.add_argument("--bot",             help="Filter to a specific bot name")
    parser.add_argument("--strategy",        help="Filter by strategy name (substring match)")
    parser.add_argument("--won",             action="store_true", help="Only winning trades")
    parser.add_argument("--lost",            action="store_true", help="Only losing trades")
    parser.add_argument("--direction",       choices=["up", "down"], help="Filter by direction")
    parser.add_argument("--candles-before",  type=int, default=20, metavar="N")
    parser.add_argument("--candles-after",   type=int, default=5,  metavar="N")
    parser.add_argument("--symbol",          default=DEFAULT_SYMBOL, help="btc or eth")
    parser.add_argument("--interval",        default=DEFAULT_INTERVAL, help="5m, 15m, 1h, 4h")
    parser.add_argument("--out",             help="Write merged CSV to this path")
    parser.add_argument("--print",           dest="print_df", action="store_true", help="Print DataFrame to stdout")
    parser.add_argument("--stats",           action="store_true", help="Print aggregate stats by strategy")
    parser.add_argument("--no-context",      action="store_true", help="Skip OHLCV join (trades only)")
    args = parser.parse_args()

    trades = load_trades(bot=args.bot)

    # Filters
    if args.strategy:
        trades = trades[trades["strategy"].str.contains(args.strategy, case=False, na=False)]
    if args.direction:
        trades = trades[trades["direction"] == args.direction]
    if args.won:
        trades = trades[trades["won"] == True]  # noqa: E712
    if args.lost:
        trades = trades[trades["won"] == False]  # noqa: E712

    print(f"Loaded {len(trades)} trades", file=sys.stderr)

    if args.stats and args.no_context:
        print_stats(trades)
        return

    if not args.no_context:
        ohlcv = load_ohlcv(symbol=args.symbol, interval=args.interval)
        print(f"OHLCV: {args.symbol}/{args.interval} — {len(ohlcv)} candles "
              f"({ohlcv['open_time'].min().date()} → {ohlcv['open_time'].max().date()})",
              file=sys.stderr)
        df = build_context(trades, ohlcv, args.candles_before, args.candles_after)
    else:
        df = trades

    if args.stats:
        print_stats(df)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows → {out}")

    if args.print_df or (not args.out and not args.stats):
        cols = [c for c in ["id", "strategy", "direction", "won", "pnl",
                             "open_time", "entry_price", "fill_price",
                             "vol_spike", "pre_ret", "entry_candle_ret"]
                if c in df.columns]
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 160)
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
