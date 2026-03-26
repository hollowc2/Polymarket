#!/usr/bin/env python3
"""Coinbase Premium Index Bot.

Trades based on the US demand signal: when Coinbase BTC-USD price diverges
significantly from Binance BTCUSDT (z-score extreme), it signals directional
bias from US institutional/retail participants.

Fetches Coinbase and Binance candles each cycle, enriches with premium index,
then evaluates the strategy signal.

Usage:
    uv run python scripts/bots/coinbase_premium_bot.py --paper --asset btc --timeframe 5m
    uv run python scripts/bots/coinbase_premium_bot.py --paper --asset btc --timeframe 15m
"""

import argparse
import os
import signal
import time
from datetime import datetime

from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.data.coinbase import compute_premium_candles, fetch_coinbase_klines
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.strategies.coinbase_premium import CoinbasePremiumStrategy

TF_SECONDS: dict[str, int] = {"5m": 300, "15m": 900}
ASSET_TO_BINANCE: dict[str, str] = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
CANDLE_BUFFER = 40  # bars for z-score window + buffer

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

    parser = argparse.ArgumentParser(description="Coinbase Premium Index Bot")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--asset", choices=["btc", "eth"], default=os.getenv("ASSET", "btc").lower())
    parser.add_argument("--timeframe", choices=["5m", "15m"], default=os.getenv("TIMEFRAME", "5m"))
    parser.add_argument("--z-thresh", type=float, default=1.5, metavar="F")
    parser.add_argument("--size", type=float, default=float(os.getenv("BET_AMOUNT", str(Config.BET_AMOUNT))))
    parser.add_argument("--bankroll", type=float)
    args = parser.parse_args()

    paper_mode = not args.live if args.live else (True if args.paper else Config.PAPER_TRADE)
    timeframe = args.timeframe
    window_seconds = TF_SECONDS[timeframe]
    asset = args.asset
    binance_symbol = ASSET_TO_BINANCE.get(asset, "BTCUSDT")

    strategy = CoinbasePremiumStrategy()
    eval_params = {"z_thresh": args.z_thresh, "size": args.size}

    client = PolymarketClient(asset=asset, timeframe=timeframe)
    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    trader = PaperTrader() if paper_mode else __import__("polymarket_algo.executor.trader", fromlist=["LiveTrader"]).LiveTrader()
    log(f"Strategy: {strategy.name} | {asset.upper()}/{timeframe} | {'PAPER' if paper_mode else 'LIVE'}")
    log(f"Params: z_thresh={args.z_thresh}, size=${args.size:.2f}")
    log(f"Timezone: {TIMEZONE_NAME} | Bankroll: ${state.bankroll:.2f}")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    pending: list = []

    while running:
        try:
            now = int(time.time())
            target_ts = ((now // window_seconds) + 1) * window_seconds
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
            candle_ms = CANDLE_BUFFER * window_seconds * 1000

            try:
                candles = fetch_klines(binance_symbol, timeframe, now_ms - candle_ms, now_ms)
                cb_candles = fetch_coinbase_klines(binance_symbol, timeframe, now_ms - candle_ms, now_ms)
            except Exception as e:
                log(f"Data fetch error: {e}")
                time.sleep(10)
                continue

            if candles.empty:
                log("No Binance candles")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            candles = candles.set_index("open_time").sort_index()
            enriched = compute_premium_candles(cb_candles, candles)

            result = strategy.evaluate(enriched, **eval_params)

            last = result.iloc[-1]
            raw_signal = int(last["signal"])
            bet_size = float(last["size"])

            if raw_signal == 0 or bet_size <= 0:
                z = enriched["coinbase_premium_zscore"].iloc[-1] if "coinbase_premium_zscore" in enriched.columns else 0.0
                log(f"No signal (premium_z={z:.2f}) — skip")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            direction = "up" if raw_signal == 1 else "down"
            z = enriched["coinbase_premium_zscore"].iloc[-1] if "coinbase_premium_zscore" in enriched.columns else 0.0
            log(f"Signal: {direction.upper()} | premium_z={z:.2f} | size=${bet_size:.2f}")

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
                gate_name="coinbase_premium",
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
