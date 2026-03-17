#!/usr/bin/env python3
"""ApexML Bot — Logistic Regression Trading on Polymarket.

Trades Polymarket crypto 5m binary markets using ApexMLStrategy:
a walk-forward logistic regression trained on multi-timeframe momentum
(15m/60m) + microstructure features.

Key finding: 15m momentum (corr_fwd=+0.22) is 8.7x stronger than 5m TFI.
Model must be pre-trained via scripts/train_apex_ml.py.

Usage
-----
    # Paper trade ETH (safe default)
    uv run python scripts/apex_ml_bot.py --paper --asset eth

    # Explore signals without placing orders
    uv run python scripts/apex_ml_bot.py --paper --asset eth --dry-run

    # Custom model path
    uv run python scripts/apex_ml_bot.py --paper --asset btc --model models/apex_ml_btc_5m.json

    # Live (requires PRIVATE_KEY in .env)
    uv run python scripts/apex_ml_bot.py --live --asset eth --max-bet 5
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from polymarket_algo.core.adapters import interpret_signal
from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.strategies.apex_ml import ApexMLStrategy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TF_SECONDS: dict[str, int] = {"5m": 300, "15m": 900}
TF_INTERVAL_MS: dict[str, int] = {"5m": 300_000, "15m": 900_000}

ASSET_TO_SYMBOL: dict[str, str] = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
}

# Warmup candles: need 300 5m bars to cover 60m features + Hawkes/OBI windows
CANDLE_WARMUP = 300

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

running = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def handle_signal(sig, _frame):
    global running
    print("\n[apex-ml] Shutting down gracefully...")
    running = False


def log(msg: str) -> None:
    ts = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fetch_enriched_candles(
    symbol: str,
    interval: str,
    end_ms: int,
) -> pd.DataFrame | None:
    """Fetch CANDLE_WARMUP closed Binance klines and derive CVD from taker_buy.

    Returns a DataFrame indexed by open_time (UTC) with columns:
        open, high, low, close, volume, buy_vol, sell_vol, delta, cvd

    Returns None on any failure.
    """
    interval_ms = TF_INTERVAL_MS.get(interval, 300_000)
    start_ms = end_ms - CANDLE_WARMUP * interval_ms

    try:
        df = fetch_klines(symbol, interval, start_ms, end_ms)
        if df.empty:
            return None
        df = df.set_index("open_time").sort_index()
        df = df[df.index < pd.Timestamp(end_ms, unit="ms", tz="UTC")]
        if df.empty:
            return None
    except Exception as exc:
        log(f"Binance candle fetch failed: {exc}")
        return None

    ohlcv = df[["open", "high", "low", "close", "volume"]].copy()

    # CVD from taker_buy columns (fast, ~1s — requires Binance with VPN)
    if "taker_buy_base_asset_volume" in df.columns:
        buy_vol = df["taker_buy_base_asset_volume"].astype(float)
        non_nan = buy_vol.notna().sum()
        if non_nan > 0:
            ohlcv["buy_vol"] = buy_vol
            ohlcv["sell_vol"] = (ohlcv["volume"] - buy_vol).clip(lower=0.0)
            ohlcv["delta"] = ohlcv["buy_vol"] - ohlcv["sell_vol"]
            ohlcv["cvd"] = ohlcv["delta"].cumsum()
        else:
            log("taker_buy columns all NaN — CVD unavailable (OKX fallback?)")

    # number_of_trades for trade_count_mtf features
    if "number_of_trades" in df.columns:
        ohlcv["number_of_trades"] = df["number_of_trades"].astype(float)

    return ohlcv


def candles_to_outcomes(candles: pd.DataFrame) -> list[str]:
    """Derive Polymarket-style outcome strings from Binance close diffs."""
    diffs = candles["close"].diff().dropna()
    return ["up" if float(d) > 0 else "down" for d in diffs]


def log_features(result: pd.DataFrame) -> None:
    """Log the last row's ApexML feature values for inspection."""
    last = result.iloc[-1]
    parts = []
    for col in ["prob_up", "edge", "signal", "size"]:
        if col in last.index:
            val = last[col]
            if isinstance(val, float):
                parts.append(f"{col}={val:+.4f}")
            else:
                parts.append(f"{col}={val}")
    log("  " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(
        description="ApexML Bot — logistic regression trading on Polymarket",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--live", action="store_true", help="Force live trading (requires PRIVATE_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate signals but do not place orders")
    parser.add_argument(
        "--asset",
        choices=["btc", "eth", "sol", "xrp"],
        default=os.getenv("ASSET", "eth").lower(),
        help="Asset to trade",
    )
    parser.add_argument(
        "--timeframe",
        choices=["5m", "15m"],
        default=os.getenv("TIMEFRAME", "5m"),
        help="Market window to target",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to trained model JSON (default: models/apex_ml_{asset}_{tf}.json)",
    )
    parser.add_argument(
        "--max-bet",
        type=float,
        metavar="USD",
        default=float(os.getenv("BET_AMOUNT", str(Config.BET_AMOUNT))),
        help="Maximum USD bet per trade (Kelly may size lower)",
    )
    parser.add_argument("--bankroll", type=float, metavar="USD", help="Override starting bankroll")
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=float(os.getenv("EDGE_THRESHOLD", "0.02")),
        metavar="FLOAT",
        help="Minimum |P(UP) - 0.5| to generate a signal",
    )
    parser.add_argument(
        "--kelly-scale",
        type=float,
        default=0.25,
        metavar="FLOAT",
        help="Fraction of full Kelly to use",
    )
    args = parser.parse_args()

    # Trading mode
    if args.live:
        paper_mode = False
    elif args.paper:
        paper_mode = True
    else:
        paper_mode = Config.PAPER_TRADE

    asset = args.asset
    timeframe = args.timeframe
    binance_symbol = ASSET_TO_SYMBOL[asset]
    window_seconds = TF_SECONDS[timeframe]
    max_bet = args.max_bet
    dry_run = args.dry_run

    # Load model
    model_path = Path(args.model) if args.model else MODELS_DIR / f"apex_ml_{asset}_{timeframe}.json"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print(f"Run: uv run python scripts/train_apex_ml.py --asset {asset} --tf {timeframe}")
        return

    strategy = ApexMLStrategy.load(model_path)
    log(f"Model loaded from {model_path}")
    log("Feature importance (top 6):")
    for feat, coef in list(strategy.feature_importance().items())[:6]:
        log(f"  {feat:12s}  {coef:+.4f}")

    eval_params: dict = {
        "edge_threshold": args.edge_threshold,
        "kelly_scale": args.kelly_scale,
        "max_bet": max_bet,
    }

    # Components
    client = PolymarketClient(asset=asset)
    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    if paper_mode:
        trader = PaperTrader()
        mode_label = "PAPER" + (" (dry-run)" if dry_run else "")
    else:
        from polymarket_algo.executor.trader import LiveTrader

        trader = LiveTrader()
        mode_label = "LIVE"

    log("=== ApexML Bot ===")
    log(f"Strategy  : {strategy.name} — {strategy.description}")
    log(f"Mode      : {mode_label}")
    log(f"Asset     : {asset.upper()} ({binance_symbol})")
    log(f"Timeframe : {timeframe}  ({window_seconds // 60}-min windows)")
    log(f"Params    : edge_threshold={args.edge_threshold} kelly={args.kelly_scale}")
    log(f"Max bet   : ${max_bet:.2f} | Bankroll: ${state.bankroll:.2f}")
    log(f"Timezone  : {TIMEZONE_NAME}")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    pending: list = []
    current_position: float = 0.0

    while running:
        try:
            now = int(time.time())
            current_5m = (now // 300) * 300
            seconds_into_5m = now - current_5m
            next_5m = current_5m + 300
            target_ts = next_5m
            seconds_until = target_ts - now

            # ── Settle pending trades ─────────────────────────────────
            for trade in list(pending):
                market = client.get_market(trade.timestamp)
                if market and market.closed and market.outcome:
                    state.settle_trade(trade, market.outcome, market)
                    pnl_sign = "+" if trade.pnl > 0 else "-"
                    settled_dir = 1 if trade.direction == "up" else -1
                    current_position -= settled_dir * float(trade.amount)
                    log(
                        f"[{pnl_sign}] Settled {trade.direction.upper()} "
                        f"@ {trade.execution_price:.3f} → {market.outcome.upper()} "
                        f"| PnL: ${trade.pnl:+.2f} | Bankroll: ${state.bankroll:.2f}"
                    )
                    pending.remove(trade)
                    state.save()

            # ── Risk limits ───────────────────────────────────────────
            can_trade, reason = state.can_trade()
            if not can_trade:
                if seconds_into_5m < 5:
                    log(f"Paused: {reason}")
                time.sleep(10)
                continue

            # Already bet this window?
            if target_ts in bet_timestamps:
                time.sleep(5)
                continue

            # Timeframe gate (15m fires every 3rd 5m window)
            if target_ts % window_seconds != 0:
                time.sleep(5)
                continue

            # ── Entry timing ──────────────────────────────────────────
            if seconds_until > Config.ENTRY_SECONDS_BEFORE:
                if seconds_into_5m % 60 == 0:
                    log(
                        f"Next {timeframe} in {seconds_until}s "
                        f"(entering at T-{Config.ENTRY_SECONDS_BEFORE}s) "
                        f"| Pending: {len(pending)} | Position: ${current_position:+.2f}"
                    )
                time.sleep(1)
                continue

            # ── Fetch enriched candles ────────────────────────────────
            end_ms = current_5m * 1000  # exclude currently-open candle
            log(f"Fetching {CANDLE_WARMUP} {timeframe} candles for {binance_symbol}…")
            candles = fetch_enriched_candles(binance_symbol, timeframe, end_ms)

            if candles is None or candles.empty:
                log("Candle fetch failed — skipping window")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            enriched_cols = [
                c for c in ["buy_vol", "sell_vol", "delta", "cvd", "number_of_trades"] if c in candles.columns
            ]
            log(f"Enriched columns: {enriched_cols or ['none — plain OHLCV']}")

            # ── Get target market ─────────────────────────────────────
            market = client.get_market(target_ts)
            if not market:
                log(f"Market not found for ts={target_ts} — skipping")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            if not market.accepting_orders:
                log(f"Market not accepting orders: {market.slug}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # ── Evaluate strategy ─────────────────────────────────────
            up_price = float(market.up_price) if market.up_price else 0.50
            down_price = float(market.down_price) if market.down_price else 0.50

            result = strategy.evaluate(candles, **eval_params)

            last = result.iloc[-1]
            raw_signal = int(last["signal"])
            raw_size = float(last["size"])

            log_features(result)

            if raw_signal == 0 or raw_size < 1.0:
                edge_val = float(last.get("edge", 0.0))
                prob_val = float(last.get("prob_up", 0.5))
                log(f"No signal — edge={edge_val:+.4f} prob_up={prob_val:.4f} threshold={args.edge_threshold}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            direction = "up" if raw_signal == 1 else "down"
            entry_price = up_price if raw_signal == 1 else down_price
            bet_size = min(raw_size, max_bet)

            log(
                f"Signal: {direction.upper()} | Size: ${bet_size:.2f} "
                f"| edge={float(last.get('edge', 0)):+.4f} "
                f"| prob_up={float(last.get('prob_up', 0.5)):.4f}"
            )

            if dry_run:
                log("[dry-run] No order placed.")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # ── Place bet ─────────────────────────────────────────────
            outcomes_for_kelly = candles_to_outcomes(candles)
            decision = interpret_signal(
                result=result,
                outcomes=outcomes_for_kelly,
                bankroll=state.bankroll,
                entry_price=entry_price,
                max_bet=bet_size,
                timeframe=timeframe,
                asset=asset.upper(),
            )

            if not decision.should_bet:
                log(f"interpret_signal veto: {decision.reason}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === BANKROLL CHECK ===
            can_trade, reason = state.can_trade(bet_size=decision.size)
            if not can_trade:
                log(f"Skipping: {reason}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            trade = trader.place_bet(
                market=market,
                direction=decision.direction,
                amount=decision.size,
                confidence=decision.confidence,
                streak_length=0,  # ML strategy, not streak-based
            )

            if trade is None:
                log("Order rejected by trader")
                bet_timestamps.add(target_ts)
                continue

            direction_sign = 1 if decision.direction == "up" else -1
            current_position += direction_sign * float(decision.size)

            state.record_trade(trade)
            bet_timestamps.add(target_ts)
            pending.append(trade)
            state.save()

            log(
                f"Order placed: {decision.direction.upper()} ${decision.size:.2f} "
                f"@ {entry_price:.3f} | Daily: {state.daily_bets} bets "
                f"PnL: ${state.daily_pnl:+.2f} | Bankroll: ${state.bankroll:.2f} "
                f"| Position: ${current_position:+.2f}"
            )

            time.sleep(5)

        except KeyboardInterrupt:
            break
        except Exception as exc:
            log(f"Error: {exc}")
            time.sleep(10)

    # ── Graceful shutdown ─────────────────────────────────────────────
    if pending:
        state.mark_pending_as_force_exit("shutdown")
    state.save()
    log(f"Shutdown complete. Bankroll: ${state.bankroll:.2f}")
    log(f"Session: {state.daily_bets} bets | PnL: ${state.daily_pnl:+.2f}")


if __name__ == "__main__":
    main()
