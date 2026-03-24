#!/usr/bin/env python3
"""Polymarket HLOrderFlow Momentum Bot — live Hyperliquid cross-asset signal.

Evaluates buy/sell pressure across Hyperliquid perpetuals (5m/15m/1h/4h) and
bets the same direction on correlated Polymarket crypto price markets.

Signal logic (handled inside HLOrderFlowMomentumStrategy):
  - HTF gate: 4h dominant_side gates allowed direction
  - Entry vote: 2-of-3 consensus across 5m, 15m, 1h (min_votes param)
  - Size: base_size × pressure_15m × 2, capped at max_size

**Live-only strategy** — HL order flow data is fetched fresh on each evaluate()
call (60s in-memory TTL). Backtesting this bot shows current state, not history.

Usage:
    uv run python scripts/hl_orderflow_bot.py --paper --asset btc --timeframe 5m
    uv run python scripts/hl_orderflow_bot.py --paper --asset eth --timeframe 15m
"""

import argparse
import os
import signal
import time
from datetime import datetime

import pandas as pd
from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.strategies.hl_orderflow_momentum import HLOrderFlowMomentumStrategy
from polymarket_algo.strategies.hl_orderflow_reversal import HLOrderFlowReversalStrategy

# Seconds per timeframe window
TF_SECONDS: dict[str, int] = {"5m": 300, "15m": 900, "1h": 3600}

# Hyperliquid coin names (uppercase)
ASSET_TO_COIN: dict[str, str] = {
    "btc": "BTC",
    "eth": "ETH",
    "sol": "SOL",
    "xrp": "XRP",
}

running = True


def handle_signal(sig, _frame):
    global running
    print("\n[bot] Shutting down gracefully...")
    running = False


def log(msg: str):
    ts = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _dummy_candles(n: int = 1) -> pd.DataFrame:
    """Minimal candle DataFrame — HLOrderFlowMomentumStrategy ignores content, uses index only."""
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=range(n),
    )


def main():
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(
        description="Polymarket HLOrderFlow Momentum Bot",
    )
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--live", action="store_true", help="Force live trading (requires PRIVATE_KEY)")
    parser.add_argument(
        "--asset",
        choices=["btc", "eth", "sol", "xrp"],
        default=os.getenv("ASSET", "btc").lower(),
        help="Crypto asset to trade (default: btc or ASSET env var)",
    )
    parser.add_argument(
        "--timeframe",
        choices=["5m", "15m", "1h"],
        default=os.getenv("TIMEFRAME", "5m"),
        help="Polling timeframe — fires once per window (default: 5m)",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=float(os.getenv("BET_AMOUNT", str(Config.BET_AMOUNT))),
        metavar="USD",
        help=f"Base bet size in USD before pressure scaling (default: {Config.BET_AMOUNT})",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=None,
        metavar="USD",
        help="Max bet cap in USD (default: 2× size)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        metavar="N",
        help="Min TF votes required for entry (default: 2)",
    )
    parser.add_argument(
        "--strategy",
        choices=["momentum", "reversal"],
        default="momentum",
        help="Signal mode: momentum (follow flow) or reversal (fade exhaustion)",
    )
    parser.add_argument("--bankroll", type=float, metavar="USD", help="Override starting bankroll")
    parser.add_argument(
        "--block-hours",
        type=str,
        default="",
        metavar="H,H,...",
        help="Comma-separated UTC hours to skip, e.g. '3,4,5,6,7,8' for reversal (default: off)",
    )

    args = parser.parse_args()

    # Trading mode — default to paper
    if args.live:
        paper_mode = False
    elif args.paper:
        paper_mode = True
    else:
        paper_mode = Config.PAPER_TRADE

    timeframe = args.timeframe
    window_seconds = TF_SECONDS[timeframe]
    asset = args.asset
    coin = ASSET_TO_COIN.get(asset, "BTC")
    base_size = args.size
    max_size = args.max_size if args.max_size is not None else base_size * 2
    min_votes = args.min_votes
    block_hours: set[int] = {int(h.strip()) for h in args.block_hours.split(",") if h.strip()}

    if args.strategy == "reversal":
        strategy = HLOrderFlowReversalStrategy()
    else:
        strategy = HLOrderFlowMomentumStrategy()
    eval_params = {
        "coin": coin,
        "size": base_size,
        "max_size": max_size,
        "min_votes": min_votes,
    }

    client = PolymarketClient(asset=asset)
    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    if paper_mode:
        trader = PaperTrader()
        log("Paper trading mode")
    else:
        from polymarket_algo.executor.trader import LiveTrader

        trader = LiveTrader()
        log("LIVE trading mode — Real money!")

    log(f"Strategy : {strategy.name}")
    log(f"Asset    : {asset.upper()} / HL coin: {coin}")
    log(f"Timeframe: {timeframe}  (fires every {window_seconds // 60} min)")
    log(f"Params   : size=${base_size:.2f}, max_size=${max_size:.2f}, min_votes={min_votes}")
    log(f"Block hrs: {sorted(block_hours) if block_hours else 'none'}")
    log(f"Max bet  : ${max_size:.2f} | Bankroll: ${state.bankroll:.2f}")
    log(f"Limits   : max {Config.MAX_DAILY_BETS} bets/day, max ${Config.MAX_DAILY_LOSS} loss/day")
    log(f"Timezone : {TIMEZONE_NAME}")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    pending: list = []

    while running:
        try:
            now = int(time.time())
            current_5m = (now // 300) * 300
            seconds_into_5m = now - current_5m
            next_5m = current_5m + 300
            target_ts = next_5m
            seconds_until_target = target_ts - now

            # === SETTLE PENDING TRADES ===
            for trade in list(pending):
                market = client.get_market(trade.timestamp)
                if market and market.closed and market.outcome:
                    state.settle_trade(trade, market.outcome, market)
                    emoji = "+" if trade.pnl > 0 else "-"
                    fee_info = f" (fee: {trade.fee_pct:.2%})" if trade.won and trade.fee_pct > 0 else ""
                    log(
                        f"[{emoji}] Settled: {trade.direction.upper()} @ {trade.execution_price:.3f} "
                        f"-> {market.outcome.upper()} | PnL: ${trade.pnl:+.2f}{fee_info} "
                        f"| Bankroll: ${state.bankroll:.2f}"
                    )
                    pending.remove(trade)
                    state.save()

            # === CHECK DAILY LIMITS ===
            can_trade, reason = state.can_trade()
            if not can_trade:
                if seconds_into_5m == 0:
                    log(f"Paused: {reason}")
                time.sleep(10)
                continue

            # Already bet this window?
            if target_ts in bet_timestamps:
                time.sleep(5)
                continue

            # === TF ALIGNMENT — only fire on TF-aligned windows ===
            if target_ts % window_seconds != 0:
                time.sleep(5)
                continue

            # === ENTRY TIMING ===
            if seconds_until_target > Config.ENTRY_SECONDS_BEFORE:
                if seconds_into_5m % 60 == 0:
                    log(
                        f"Next {timeframe} window in {seconds_until_target}s "
                        f"(entering at T-{Config.ENTRY_SECONDS_BEFORE}s) | "
                        f"Pending: {len(pending)} trades"
                    )
                time.sleep(1)
                continue

            # === SESSION BLOCK ===
            target_hour = (target_ts % 86400) // 3600
            if block_hours and target_hour in block_hours:
                log(f"Session block: hour {target_hour} UTC — skip")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === EVALUATE STRATEGY ===
            log(f"Fetching HL order flow for {coin}...")
            result = strategy.evaluate(_dummy_candles(), **eval_params)

            last_row = result.iloc[-1]
            raw_signal = int(last_row["signal"])
            bet_size = float(last_row["size"])

            if raw_signal == 0 or bet_size <= 0:
                log(f"No signal (signal={raw_signal}, size={bet_size:.2f}) — skip")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            direction = "up" if raw_signal == 1 else "down"
            log(f"Signal: {direction.upper()}  size=${bet_size:.2f}")

            # === BANKROLL CHECK ===
            can_trade, reason = state.can_trade(bet_size=bet_size)
            if not can_trade:
                log(f"Skipping: {reason}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === GET TARGET MARKET ===
            market = client.get_market(target_ts)
            if not market:
                log(f"Market not found for ts={target_ts}")
                time.sleep(5)
                continue

            if not market.accepting_orders:
                log(f"Market not accepting orders: {market.slug}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === PLACE BET ===
            trade = trader.place_bet(
                market=market,
                direction=direction,
                amount=bet_size,
                confidence=1.0,
                streak_length=0,
                gate_name="htf_4h",
                gate_boosted=False,
                gate_skipped=False,
            )

            if trade is None:
                log("Order rejected")
                bet_timestamps.add(target_ts)
                continue

            state.record_trade(trade)
            bet_timestamps.add(target_ts)
            pending.append(trade)
            state.save()

            log(
                f"Placed: {direction.upper()} ${bet_size:.2f} on {market.slug} "
                f"| Daily: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f} "
                f"| Bankroll: ${state.bankroll:.2f}"
            )

            time.sleep(5)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(10)

    # Graceful shutdown
    if pending:
        state.mark_pending_as_force_exit("shutdown")
    state.save()
    log(f"State saved. Bankroll: ${state.bankroll:.2f}")
    log(f"Session: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f}")


if __name__ == "__main__":
    main()
