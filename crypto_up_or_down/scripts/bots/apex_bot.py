#!/usr/bin/env python3
"""APEX Bot — Streak Reversal + Microstructure Exhaustion Trading on Polymarket.

Trades Polymarket crypto 5m/15m binary markets (BTC, ETH, SOL, XRP) using
ApexHybridStrategy: streak reversal gated by TFI flow exhaustion signals.

Walk-forward validated results (train 2022-2023 / test 2024-2026):
    ETH/5m:  Sharpe 6.54, win 55.5%, ~11k trades/2yr
    ETH/15m: Sharpe 5.62, win 55.8%, ~7.7k trades/2yr
    BTC/15m: Sharpe 6.21, win 56.1%, ~7.5k trades/2yr
    BTC/5m:  Sharpe 3.84, win 54.3%, ~11k trades/2yr

Every market window the bot:
  1. Fetches the last CANDLE_WARMUP real Binance OHLCV + CVD candles
  2. Evaluates ApexHybridStrategy → streak signal filtered by TFI exhaustion
  3. Places a market order via PaperTrader / LiveTrader
  4. Polls for fills and settles completed markets

Usage
-----
    # Paper trade (safe default)
    uv run python scripts/apex_bot.py --paper --asset eth

    # Live trade (requires PRIVATE_KEY in .env)
    uv run python scripts/apex_bot.py --live --asset btc --max-bet 5

    # Explore signals without placing orders
    uv run python scripts/apex_bot.py --paper --asset btc --dry-run

Configuration via CLI args or environment variables:
    ASSET              Asset (btc/eth/sol/xrp) — default btc
    TIMEFRAME          5m or 15m               — default 5m
    BET_AMOUNT         Max USD bet per trade    — default 15.0
    PAPER_TRADE        true/false               — default true
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from datetime import datetime

import pandas as pd
from polymarket_algo.core.adapters import (
    interpret_signal,
)
from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.strategies.apex_hybrid import ApexHybridStrategy

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

# Warmup candles for Hawkes/EMA/OBI windows (covers 200-bar windows + buffer)
CANDLE_WARMUP = 300

# Minimum fill probability before we use limit orders
MIN_FILL_PROB_FOR_LIMIT = 0.60

running = True
strategy = ApexHybridStrategy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def handle_signal(sig, _frame):
    global running
    print("\n[apex] Shutting down gracefully...")
    running = False


def log(msg: str) -> None:
    ts = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fetch_enriched_candles(
    symbol: str,
    interval: str,
    end_ms: int,
) -> pd.DataFrame | None:
    """Fetch CANDLE_WARMUP closed Binance klines and derive CVD from taker_buy columns.

    CVD is computed directly from the kline taker_buy_base_asset_volume column
    (available when fetching from Binance with VPN). This is fast (~1s) vs
    enrich_candles() which fetches full agg_trades (~4+ minutes for 300 bars).

    Returns a DataFrame indexed by open_time (UTC) with columns:
        open, high, low, close, volume, buy_vol, sell_vol, delta, cvd

    Returns None on any failure so the caller can skip this window.
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

    # Derive CVD from taker_buy columns (fast — no extra API call)
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

    return ohlcv


def candles_to_outcomes(candles: pd.DataFrame) -> list[str]:
    """Derive Polymarket-style outcome strings from real Binance close diffs."""
    diffs = candles["close"].diff().dropna()
    return ["up" if float(d) > 0 else "down" for d in diffs]


def log_features(result: pd.DataFrame) -> None:
    """Log the last row's APEX feature values for inspection."""
    last = result.iloc[-1]
    diag_cols = [
        "streak",
        "streak_dir",
        "tfi_exhaust",
        "mp_exhaust",
        "exhaust_score",
        "hawkes_intensity",
        "regime_ok",
    ]
    parts = []
    for col in diag_cols:
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
        description="APEX Bot — probabilistic microstructure trading on Polymarket",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--live", action="store_true", help="Force live trading (requires PRIVATE_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate signals but do not place any orders")
    parser.add_argument(
        "--asset",
        choices=["btc", "eth", "sol", "xrp"],
        default=os.getenv("ASSET", "btc").lower(),
        help="Asset to trade",
    )
    parser.add_argument(
        "--timeframe",
        choices=["5m", "15m"],
        default=os.getenv("TIMEFRAME", "5m"),
        help="Market window to target",
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
        "--trigger",
        type=int,
        default=4,
        metavar="N",
        help="Streak trigger length (default: 4)",
    )
    parser.add_argument(
        "--kelly-scale",
        type=float,
        default=0.25,
        metavar="FLOAT",
        help="Fraction of full Kelly to use (default: 0.25)",
    )
    parser.add_argument(
        "--w-tfi-exhaust",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="TFI exhaustion weight (default: 1.0; 0 = disabled)",
    )
    parser.add_argument(
        "--exhaust-mult",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Exhaustion size boost multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--hawkes-threshold",
        type=float,
        default=0.53,
        metavar="FLOAT",
        help="Hawkes activity gate threshold (default: 0.53; 0 = off)",
    )
    parser.add_argument(
        "--regime",
        action="store_true",
        default=False,
        help="Enable ATR/volume regime gate (off by default — too strict in quiet live markets)",
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

    eval_params: dict = {
        "trigger": args.trigger,
        "kelly_scale": args.kelly_scale,
        "w_tfi_exhaust": args.w_tfi_exhaust,
        "exhaust_mult": args.exhaust_mult,
        "hawkes_threshold": args.hawkes_threshold,
        "use_regime": args.regime,
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

    log("=== APEX Hybrid Bot ===")
    log(f"Strategy  : {strategy.name} — {strategy.description}")
    log(f"Mode      : {mode_label}")
    log(f"Asset     : {asset.upper()} ({binance_symbol})")
    log(f"Timeframe : {timeframe}  ({window_seconds // 60}-min windows)")
    log(
        f"Params    : trigger={args.trigger} tfi_exhaust={args.w_tfi_exhaust} "
        f"exhaust_mult={args.exhaust_mult} hawkes_gate={args.hawkes_threshold}"
    )
    log(f"Max bet   : ${max_bet:.2f} | Kelly scale: {args.kelly_scale:.2f}")
    log(f"Bankroll  : ${state.bankroll:.2f}")
    log(f"Timezone  : {TIMEZONE_NAME}")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    pending: list = []
    current_position: float = 0.0  # running net USD exposure (+ = long, - = short)

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
                    # Update position tracking on settlement
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

            # Log which enriched columns are present
            enriched_cols = [
                c
                for c in ["buy_vol", "sell_vol", "delta", "cvd", "liq_long_usd", "liq_short_usd", "funding_zscore"]
                if c in candles.columns
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
                streak_val = int(last.get("streak", 0))
                log(
                    f"No signal — streak={streak_val} "
                    f"exhaust={float(last.get('exhaust_score', 0)):.3f} "
                    f"hawkes={float(last.get('hawkes_intensity', 0)):.3f} "
                    f"regime={int(last.get('regime_ok', 0))}"
                )
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            direction = "up" if raw_signal == 1 else "down"
            entry_price = up_price if raw_signal == 1 else down_price
            bet_size = min(raw_size, max_bet)

            log(
                f"Signal: {direction.upper()} | Size: ${bet_size:.2f} "
                f"| streak={int(last.get('streak', 0))} "
                f"| exhaust={float(last.get('exhaust_score', 0)):.3f} "
                f"| hawkes={float(last.get('hawkes_intensity', 0)):.3f}"
            )

            if dry_run:
                log("[dry-run] No order placed.")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # ── Place bet ─────────────────────────────────────────────
            # Use interpret_signal for timeframe-aware confidence routing
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
                streak_length=0,  # APEX is not streak-based
            )

            if trade is None:
                log("Order rejected by trader")
                bet_timestamps.add(target_ts)
                continue

            # Track running position
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
