#!/usr/bin/env python3
"""Cross-Timeframe Consistency Bot.

Monitors Polymarket 5m, 15m, and 60m YES/NO markets simultaneously and trades
the shorter-timeframe market when the probabilities are inconsistent with the
longer-timeframe market.

  60m UP >> 5m UP → 5m market underpriced → bet UP on 5m
  60m DOWN >> 5m DOWN → 5m market overpriced → bet DOWN on 5m

Fetches YES prices from Polymarket REST for all three timeframes each cycle.
Trades the 5m market (most frequent, highest liquidity).

Usage:
    uv run python scripts/bots/cross_tf_bot.py --paper --asset btc
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
from polymarket_algo.strategies.cross_tf_consistency import CrossTFConsistencyStrategy

ASSET_TO_SYMBOL: dict[str, str] = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
WINDOW_5M = 300
WINDOW_15M = 900
WINDOW_60M = 3600
CANDLE_BUFFER = 5  # just need a few recent candles for index

running = True


def handle_signal(sig, _frame):
    global running
    print("\n[bot] Shutting down gracefully...")
    running = False


def log(msg: str):
    ts = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_next_window_ts(window_seconds: int) -> int:
    """Get the closing timestamp of the next window for a given interval."""
    now = int(time.time())
    return ((now // window_seconds) + 1) * window_seconds


def seconds_until(ts: int) -> int:
    return ts - int(time.time())


def main():
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(description="Cross-Timeframe Consistency Bot")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--asset", choices=["btc", "eth"], default=os.getenv("ASSET", "btc").lower())
    parser.add_argument("--consistency-thresh", type=float, default=0.08, metavar="F",
                        help="Min |gap| between 5m and 60m YES prices to trigger trade")
    parser.add_argument("--require-15m-confirm", action="store_true",
                        help="Also check that 15m aligns with 60m direction")
    parser.add_argument("--size", type=float, default=float(os.getenv("BET_AMOUNT", str(Config.BET_AMOUNT))))
    parser.add_argument("--bankroll", type=float)
    args = parser.parse_args()

    paper_mode = not args.live if args.live else (True if args.paper else Config.PAPER_TRADE)
    asset = args.asset
    symbol = ASSET_TO_SYMBOL.get(asset, "BTCUSDT")

    strategy = CrossTFConsistencyStrategy()
    eval_params = {
        "consistency_thresh": args.consistency_thresh,
        "require_15m_confirm": args.require_15m_confirm,
        "size": args.size,
    }

    # Create clients for all three timeframes
    client_5m = PolymarketClient(asset=asset, timeframe="5m")
    client_15m = PolymarketClient(asset=asset, timeframe="15m")
    client_60m = PolymarketClient(asset=asset, timeframe="1h")

    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    trader = PaperTrader() if paper_mode else __import__("polymarket_algo.executor.trader", fromlist=["LiveTrader"]).LiveTrader()
    log(f"Strategy: {strategy.name} | {asset.upper()}/5m (cross-TF) | {'PAPER' if paper_mode else 'LIVE'}")
    log(f"Params: consistency_thresh={args.consistency_thresh}, require_15m={args.require_15m_confirm}, "
        f"size=${args.size:.2f}")
    log(f"Timezone: {TIMEZONE_NAME} | Bankroll: ${state.bankroll:.2f}")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    pending: list = []

    while running:
        try:
            target_5m = get_next_window_ts(WINDOW_5M)
            secs_to_5m = seconds_until(target_5m)

            # Settle pending trades
            for trade in list(pending):
                market = client_5m.get_market(trade.timestamp)
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

            if target_5m in bet_timestamps:
                time.sleep(5)
                continue

            if secs_to_5m > Config.ENTRY_SECONDS_BEFORE:
                time.sleep(1)
                continue

            # Fetch YES prices for all three timeframes via REST
            target_15m = get_next_window_ts(WINDOW_15M)
            target_60m = get_next_window_ts(WINDOW_60M)

            yes_5m: float | None = None
            yes_15m: float | None = None
            yes_60m: float | None = None

            try:
                m5 = client_5m.get_market(target_5m)
                if m5 and m5.up_token_id:
                    yes_5m = client_5m.get_midpoint(m5.up_token_id)
            except Exception as e:
                log(f"5m price fetch error: {e}")

            try:
                m15 = client_15m.get_market(target_15m)
                if m15 and m15.up_token_id:
                    yes_15m = client_15m.get_midpoint(m15.up_token_id)
            except Exception as e:
                log(f"15m price fetch error: {e}")

            try:
                m60 = client_60m.get_market(target_60m)
                if m60 and m60.up_token_id:
                    yes_60m = client_60m.get_midpoint(m60.up_token_id)
            except Exception as e:
                log(f"60m price fetch error: {e}")

            log(f"YES prices — 5m: {yes_5m}, 15m: {yes_15m}, 60m: {yes_60m}")

            if yes_5m is None or yes_60m is None:
                log("Missing YES prices — skipping")
                bet_timestamps.add(target_5m)
                time.sleep(5)
                continue

            # Fetch a minimal candle DF just for the strategy index
            now_ms = int(time.time() * 1000)
            try:
                candles = fetch_klines(symbol, "5m", now_ms - CANDLE_BUFFER * WINDOW_5M * 1000, now_ms)
                if candles.empty:
                    raise ValueError("No candles")
                candles = candles.set_index("open_time").sort_index()
            except Exception as e:
                log(f"Candle fetch error: {e}")
                bet_timestamps.add(target_5m)
                time.sleep(5)
                continue

            result = strategy.evaluate(
                candles,
                yes_price_5m=yes_5m,
                yes_price_60m=yes_60m,
                yes_price_15m=yes_15m,
                **eval_params,
            )

            last = result.iloc[-1]
            raw_signal = int(last["signal"])
            bet_size = float(last["size"])

            gap = (yes_60m or 0.5) - (yes_5m or 0.5)
            if raw_signal == 0 or bet_size <= 0:
                log(f"No inconsistency (gap={gap:+.3f}) — skip")
                bet_timestamps.add(target_5m)
                time.sleep(5)
                continue

            direction = "up" if raw_signal == 1 else "down"
            log(f"Signal: {direction.upper()} | gap={gap:+.3f} | size=${bet_size:.2f}")

            can_trade, reason = state.can_trade(bet_size=bet_size)
            if not can_trade:
                log(f"Skipping: {reason}")
                bet_timestamps.add(target_5m)
                time.sleep(5)
                continue

            market = client_5m.get_market(target_5m)
            if not market or not market.accepting_orders:
                log(f"5m market unavailable for ts={target_5m}")
                bet_timestamps.add(target_5m)
                time.sleep(5)
                continue

            trade = trader.place_bet(
                market=market,
                direction=direction,
                amount=bet_size,
                confidence=0.55,
                streak_length=0,
                gate_name="cross_tf_consistency",
            )

            if trade is None:
                log("Order rejected")
                bet_timestamps.add(target_5m)
                continue

            state.record_trade(trade)
            bet_timestamps.add(target_5m)
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
