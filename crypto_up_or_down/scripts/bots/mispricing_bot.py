#!/usr/bin/env python3
"""Polymarket Fair Value Mispricing Bot.

Compares Polymarket's live YES token price to a fair-value estimate derived
from Binance OHLCV historical base rate. Trades when the market price deviates
significantly from the fair value.

  YES too cheap (< fair_value - thresh) → bet UP
  YES too expensive (> fair_value + thresh) → bet DOWN

Fetches:
  - Binance OHLCV history for fair value computation
  - Polymarket REST API for live YES token price

Usage:
    uv run python scripts/bots/mispricing_bot.py --paper --asset btc
"""

import argparse
import os
import signal
import time
from datetime import datetime

from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.strategies.polymarket_mispricing import PolymarketMispricingStrategy

ASSET_TO_SYMBOL: dict[str, str] = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
# Need enough bars for the lookback window (default 500 bars)
CANDLE_BUFFER = 600
WINDOW_SECONDS = 300  # 5m only

running = True


def handle_signal(sig, _frame):
    global running
    print("\n[bot] Shutting down gracefully...")
    running = False


def log(msg: str):
    ts = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(description="Polymarket Fair Value Mispricing Bot")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--asset", choices=["btc", "eth"], default=os.getenv("ASSET", "btc").lower())
    parser.add_argument("--lookback", type=int, default=500, metavar="N",
                        help="Historical base rate window in bars")
    parser.add_argument("--recent-window", type=int, default=20, metavar="N")
    parser.add_argument("--recent-weight", type=float, default=0.3, metavar="F")
    parser.add_argument("--misprice-thresh", type=float, default=0.04, metavar="F",
                        help="Minimum gap (in prob units) to trigger a trade")
    parser.add_argument("--size", type=float, default=float(os.getenv("BET_AMOUNT", str(Config.BET_AMOUNT))))
    parser.add_argument("--bankroll", type=float)
    args = parser.parse_args()

    paper_mode = not args.live if args.live else (True if args.paper else Config.PAPER_TRADE)
    asset = args.asset
    symbol = ASSET_TO_SYMBOL.get(asset, "BTCUSDT")

    strategy = PolymarketMispricingStrategy()
    eval_params = {
        "lookback": args.lookback,
        "recent_window": args.recent_window,
        "recent_weight": args.recent_weight,
        "misprice_thresh": args.misprice_thresh,
        "size": args.size,
    }

    client = PolymarketClient(asset=asset)
    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    trader = PaperTrader() if paper_mode else __import__("polymarket_algo.executor.trader", fromlist=["LiveTrader"]).LiveTrader()
    log(f"Strategy: {strategy.name} | {asset.upper()}/5m | {'PAPER' if paper_mode else 'LIVE'}")
    log(f"Params: lookback={args.lookback}, recent_weight={args.recent_weight}, "
        f"misprice_thresh={args.misprice_thresh}, size=${args.size:.2f}")
    log(f"Timezone: {TIMEZONE_NAME} | Bankroll: ${state.bankroll:.2f}")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    pending: list = []

    while running:
        try:
            now = int(time.time())
            target_ts = ((now // WINDOW_SECONDS) + 1) * WINDOW_SECONDS
            seconds_until_target = target_ts - now

            for trade in list(pending):
                market = client.get_market(trade.timestamp)
                if market and market.closed and market.outcome:
                    state.settle_trade(trade, market.outcome, market)
                    emoji = "+" if trade.pnl > 0 else "-"
                    log(f"[{emoji}] Settled: {trade.direction.upper()} -> {market.outcome.upper()} "
                        f"| PnL: ${trade.pnl:+.2f} | Bankroll: ${state.bankroll:.2f}")
                    pending.remove(trade)
                    state.save()

            can_trade, reason = state.can_trade()
            if not can_trade:
                time.sleep(10)
                continue

            if target_ts in bet_timestamps:
                time.sleep(5)
                continue

            if seconds_until_target > Config.ENTRY_SECONDS_BEFORE:
                time.sleep(1)
                continue

            now_ms = int(time.time() * 1000)
            candle_ms = CANDLE_BUFFER * WINDOW_SECONDS * 1000

            try:
                candles = fetch_klines(symbol, "5m", now_ms - candle_ms, now_ms)
            except Exception as e:
                log(f"Binance fetch error: {e}")
                time.sleep(10)
                continue

            if candles.empty or len(candles) < 50:
                log(f"Insufficient candles: {len(candles)}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            candles = candles.set_index("open_time").sort_index()

            # Fetch live Polymarket YES price
            yes_price: float | None = None
            try:
                market_info = client.get_market(target_ts)
                if market_info and market_info.up_token_id:
                    yes_price = client.get_midpoint(market_info.up_token_id)
            except Exception as e:
                log(f"Polymarket price fetch error: {e}")

            if yes_price is None:
                log("Cannot fetch YES price — skipping")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            result = strategy.evaluate(candles, yes_price=yes_price, **eval_params)

            last = result.iloc[-1]
            raw_signal = int(last["signal"])
            bet_size = float(last["size"])

            if raw_signal == 0 or bet_size <= 0:
                log(f"No misprice (YES={yes_price:.3f}) — skip")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            direction = "up" if raw_signal == 1 else "down"
            log(f"Signal: {direction.upper()} | YES={yes_price:.3f} | size=${bet_size:.2f}")

            can_trade, reason = state.can_trade(bet_size=bet_size)
            if not can_trade:
                log(f"Skipping: {reason}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            market = client.get_market(target_ts)
            if not market or not market.accepting_orders:
                log(f"Market unavailable for ts={target_ts}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            trade = trader.place_bet(
                market=market,
                direction=direction,
                amount=bet_size,
                confidence=0.55,
                streak_length=0,
                gate_name="polymarket_mispricing",
            )

            if trade is None:
                log("Order rejected")
                bet_timestamps.add(target_ts)
                continue

            state.record_trade(trade)
            bet_timestamps.add(target_ts)
            pending.append(trade)
            state.save()
            log(f"Placed: {direction.upper()} ${bet_size:.2f} on {market.slug} "
                f"| Daily: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f}")

            time.sleep(5)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(10)

    if pending:
        state.mark_pending_as_force_exit("shutdown")
    state.save()
    log(f"State saved. Bankroll: ${state.bankroll:.2f} | Session: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f}")


if __name__ == "__main__":
    main()
