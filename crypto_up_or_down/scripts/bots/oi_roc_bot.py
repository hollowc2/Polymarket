#!/usr/bin/env python3
"""Open Interest Rate-of-Change Bot.

Detects squeeze setups (OI spike + price flat) and unwinding signals
(OI collapse + price flat) to generate directional bets.

Fetches OI history, spot OHLCV, and funding rate each cycle.

Usage:
    uv run python scripts/bots/oi_roc_bot.py --paper --asset btc
"""

import argparse
import os
import signal
import time
from datetime import datetime

from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.data.binance import compute_oi_candles, fetch_klines, fetch_open_interest_hist
from polymarket_algo.data.funding import compute_funding_candles, fetch_funding_rate
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.strategies.oi_roc import OIRateOfChangeStrategy

ASSET_TO_SYMBOL: dict[str, str] = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
CANDLE_BUFFER = 40

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

    parser = argparse.ArgumentParser(description="Open Interest Rate-of-Change Bot")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--asset", choices=["btc", "eth"], default=os.getenv("ASSET", "btc").lower())
    parser.add_argument("--oi-z-thresh", type=float, default=2.0, metavar="F")
    parser.add_argument("--price-flat-thresh", type=float, default=0.001, metavar="F")
    parser.add_argument("--size", type=float, default=float(os.getenv("BET_AMOUNT", str(Config.BET_AMOUNT))))
    parser.add_argument("--bankroll", type=float)
    args = parser.parse_args()

    paper_mode = not args.live if args.live else (True if args.paper else Config.PAPER_TRADE)
    asset = args.asset
    symbol = ASSET_TO_SYMBOL.get(asset, "BTCUSDT")
    window_seconds = 300  # 5m only

    strategy = OIRateOfChangeStrategy()
    eval_params = {
        "oi_z_thresh": args.oi_z_thresh,
        "price_flat_thresh": args.price_flat_thresh,
        "size": args.size,
    }

    client = PolymarketClient(asset=asset, timeframe="5m")
    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    trader = PaperTrader() if paper_mode else __import__("polymarket_algo.executor.trader", fromlist=["LiveTrader"]).LiveTrader()
    log(f"Strategy: {strategy.name} | {asset.upper()}/5m | {'PAPER' if paper_mode else 'LIVE'}")
    log(f"Params: oi_z={args.oi_z_thresh}, price_flat={args.price_flat_thresh}, size=${args.size:.2f}")
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
            start_ms = now_ms - candle_ms

            try:
                candles = fetch_klines(symbol, "5m", start_ms, now_ms)
                oi_df = fetch_open_interest_hist(symbol, "5m", start_ms, now_ms)
                funding_df = fetch_funding_rate(symbol, start_ms, now_ms)
            except Exception as e:
                log(f"Data fetch error: {e}")
                time.sleep(10)
                continue

            if candles.empty:
                log("No candles")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            candles = candles.set_index("open_time").sort_index()
            enriched = compute_oi_candles(oi_df, candles)
            enriched = compute_funding_candles(funding_df, enriched)

            result = strategy.evaluate(enriched, **eval_params)

            last = result.iloc[-1]
            raw_signal = int(last["signal"])
            bet_size = float(last["size"])

            if raw_signal == 0 or bet_size <= 0:
                oi_z = enriched["oi_zscore"].iloc[-1] if "oi_zscore" in enriched.columns else 0.0
                log(f"No signal (oi_z={oi_z:.2f}) — skip")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            direction = "up" if raw_signal == 1 else "down"
            oi_z = enriched["oi_zscore"].iloc[-1] if "oi_zscore" in enriched.columns else 0.0
            log(f"Signal: {direction.upper()} | oi_z={oi_z:.2f} | size=${bet_size:.2f}")

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
                gate_name="oi_roc",
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
