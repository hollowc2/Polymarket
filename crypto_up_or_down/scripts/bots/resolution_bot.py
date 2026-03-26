#!/usr/bin/env python3
"""Resolution Convergence Bot.

In the final seconds of a Polymarket 5m window, the YES/NO token price should
converge toward 0 or 1. When the Binance price has clearly committed above or
below the market's reference open price but Polymarket hasn't priced this in
yet, a near-certain edge exists.

Fires significantly earlier than other bots (--trigger-seconds before window
close, default 45s) to catch the convergence window.

Requires:
  - Binance real-time price (last close from recent candle)
  - Polymarket YES price (from REST client)
  - Market open price (Binance price at the start of this 5m window)

Usage:
    uv run python scripts/bots/resolution_bot.py --paper --asset btc
"""

import argparse
import os
import signal
import time
from datetime import UTC, datetime

from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.strategies.resolution_convergence import ResolutionConvergenceStrategy

ASSET_TO_SYMBOL: dict[str, str] = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
WINDOW_SECONDS = 300  # 5m only
# Fetch just a few recent candles — only need current and open prices
CANDLE_BUFFER = 5

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

    parser = argparse.ArgumentParser(description="Resolution Convergence Bot")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--asset", choices=["btc", "eth"], default=os.getenv("ASSET", "btc").lower())
    parser.add_argument("--trigger-seconds", type=int, default=45, metavar="N",
                        help="Activate in last N seconds of window")
    parser.add_argument("--yes-ceiling", type=float, default=0.85, metavar="F",
                        help="UP outcome: YES cheap if below this")
    parser.add_argument("--no-floor", type=float, default=0.15, metavar="F",
                        help="DOWN outcome: NO cheap if YES above this")
    parser.add_argument("--price-margin", type=float, default=0.0001, metavar="F",
                        help="Min fractional move from open to consider 'committed'")
    parser.add_argument("--size", type=float, default=float(os.getenv("BET_AMOUNT", str(Config.BET_AMOUNT))))
    parser.add_argument("--bankroll", type=float)
    args = parser.parse_args()

    paper_mode = not args.live if args.live else (True if args.paper else Config.PAPER_TRADE)
    asset = args.asset
    symbol = ASSET_TO_SYMBOL.get(asset, "BTCUSDT")

    strategy = ResolutionConvergenceStrategy()

    client = PolymarketClient(asset=asset, timeframe="5m")
    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    trader = PaperTrader() if paper_mode else __import__("polymarket_algo.executor.trader", fromlist=["LiveTrader"]).LiveTrader()
    log(f"Strategy: {strategy.name} | {asset.upper()}/5m | {'PAPER' if paper_mode else 'LIVE'}")
    log(f"Params: trigger_s={args.trigger_seconds}, yes_ceil={args.yes_ceiling}, "
        f"no_floor={args.no_floor}, size=${args.size:.2f}")
    log(f"Timezone: {TIMEZONE_NAME} | Bankroll: ${state.bankroll:.2f}")
    log("Note: fires at T-trigger_seconds, not T-ENTRY_SECONDS_BEFORE")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    pending: list = []

    # Track the market open price (Binance at window start)
    current_window_ts: int = 0
    market_open_price: float | None = None

    while running:
        try:
            now = int(time.time())
            current_ts = (now // WINDOW_SECONDS) * WINDOW_SECONDS  # start of current window
            target_ts = current_ts + WINDOW_SECONDS  # end / resolution time
            seconds_remaining = target_ts - now

            # At the start of a new window, record the current Binance price as "open"
            if current_ts != current_window_ts:
                current_window_ts = current_ts
                market_open_price = None
                try:
                    now_ms = int(time.time() * 1000)
                    candles = fetch_klines(symbol, "5m", now_ms - CANDLE_BUFFER * WINDOW_SECONDS * 1000, now_ms)
                    if not candles.empty:
                        # Open price of the CURRENT candle (most recent open)
                        candles = candles.set_index("open_time").sort_index()
                        market_open_price = float(candles["open"].iloc[-1])
                        log(f"New window {datetime.fromtimestamp(current_ts, tz=UTC).strftime('%H:%M:%S')} "
                            f"| open_price={market_open_price:.2f}")
                except Exception as e:
                    log(f"Open price fetch error: {e}")

            # Settle pending trades
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
                time.sleep(5)
                continue

            if target_ts in bet_timestamps:
                time.sleep(2)
                continue

            # Only fire within the trigger window
            if seconds_remaining > args.trigger_seconds:
                time.sleep(1)
                continue

            if market_open_price is None:
                log("No open price recorded — skipping this window")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # Fetch current Binance price
            try:
                now_ms = int(time.time() * 1000)
                candles = fetch_klines(symbol, "5m", now_ms - CANDLE_BUFFER * WINDOW_SECONDS * 1000, now_ms)
                if candles.empty:
                    raise ValueError("Empty candles")
                candles = candles.set_index("open_time").sort_index()
            except Exception as e:
                log(f"Candle fetch error: {e}")
                time.sleep(5)
                continue

            # Fetch live YES price
            yes_price: float | None = None
            try:
                market_info = client.get_market(target_ts)
                if market_info and market_info.up_token_id:
                    yes_price = client.get_midpoint(market_info.up_token_id)
            except Exception as e:
                log(f"YES price fetch error: {e}")

            if yes_price is None:
                log("Cannot fetch YES price")
                time.sleep(5)
                continue

            eval_params = {
                "trigger_seconds": args.trigger_seconds,
                "price_margin": args.price_margin,
                "yes_ceiling": args.yes_ceiling,
                "no_floor": args.no_floor,
                "yes_price": yes_price,
                "seconds_remaining": seconds_remaining,
                "market_open_price": market_open_price,
                "size": args.size,
            }

            result = strategy.evaluate(candles, **eval_params)
            last = result.iloc[-1]
            raw_signal = int(last["signal"])
            bet_size = float(last["size"])

            current_price = float(candles["close"].iloc[-1])
            log(f"T-{seconds_remaining}s | price={current_price:.2f} vs open={market_open_price:.2f} "
                f"| YES={yes_price:.3f} | signal={raw_signal}")

            if raw_signal == 0 or bet_size <= 0:
                bet_timestamps.add(target_ts)
                time.sleep(2)
                continue

            direction = "up" if raw_signal == 1 else "down"

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

            log(f"FIRING: {direction.upper()} | T-{seconds_remaining}s | YES={yes_price:.3f} | ${bet_size:.2f}")
            trade = trader.place_bet(
                market=market,
                direction=direction,
                amount=bet_size,
                confidence=0.65,
                streak_length=0,
                gate_name="resolution_convergence",
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

            time.sleep(2)

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
