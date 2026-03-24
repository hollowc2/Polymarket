#!/usr/bin/env python3
"""Polymarket Delta Flip Bot — zero-crossing signal on Hyperliquid cumulative delta.

Fires a directional bet the first candle after cumulative_delta (buy_vol - sell_vol)
crosses zero on the signal timeframe, optionally gated by 4h dominant_side.

Signal logic (handled inside DeltaFlipStrategy):
  - Zero-crossing: prev_delta * cur_delta < 0 → regime flip detected
  - HTF gate: 4h dominant_side vetoes flips against the higher-timeframe trend
  - Size: flat min(base_size, max_size) — binary flip, no pressure scalar

**Live-only strategy** — cumulative delta is fetched fresh each cycle (60s TTL).

Usage:
    uv run python scripts/delta_flip_bot.py --paper --asset btc
    uv run python scripts/delta_flip_bot.py --paper --asset eth --timeframe 15m
    uv run python scripts/delta_flip_bot.py --paper --asset btc --gate-timeframe 1h
"""

import argparse
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.strategies.delta_flip import DeltaFlipStrategy

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
    """Minimal candle DataFrame — DeltaFlipStrategy ignores content, uses index only."""
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=range(n),
    )


def main():
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(description="Polymarket Delta Flip Bot")
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--live", action="store_true", help="Force live trading (requires PRIVATE_KEY)")
    parser.add_argument(
        "--asset",
        choices=["btc", "eth", "sol", "xrp"],
        default=os.getenv("ASSET", "btc").lower(),
        help="Crypto asset to trade (default: btc or ASSET env var)",
    )
    parser.add_argument(
        "--coin",
        default=None,
        help="Override Hyperliquid coin (default: derived from --asset)",
    )
    parser.add_argument(
        "--timeframe",
        choices=["5m", "15m", "1h"],
        default=os.getenv("TIMEFRAME", "5m"),
        help="Signal timeframe — fires once per window (default: 5m)",
    )
    parser.add_argument(
        "--gate-timeframe",
        choices=["1h", "4h"],
        default="1h",
        help="HTF gate timeframe (default: 1h)",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=float(os.getenv("BET_AMOUNT", str(Config.BET_AMOUNT))),
        metavar="USD",
        help=f"Bet size in USD (default: {Config.BET_AMOUNT})",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=None,
        metavar="USD",
        help="Max bet cap in USD (default: 5× size)",
    )
    parser.add_argument("--bankroll", type=float, metavar="USD", help="Override starting bankroll")
    parser.add_argument(
        "--block-hours",
        type=str,
        default="0,1",
        metavar="H,H,...",
        help="Comma-separated UTC hours to skip (default: 0,1)",
    )
    parser.add_argument(
        "--vol-confirm",
        action="store_true",
        help="Require current bar volume > rolling median before firing signal",
    )
    parser.add_argument(
        "--vol-confirm-window",
        type=int,
        default=20,
        metavar="N",
        help="Rolling window (bars) for volume median check (default: 20)",
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
    coin = args.coin if args.coin else ASSET_TO_COIN.get(asset, "BTC")
    base_size = args.size
    max_size = args.max_size if args.max_size is not None else base_size * 5
    gate_timeframe = args.gate_timeframe
    block_hours: set[int] = {int(h.strip()) for h in args.block_hours.split(",") if h.strip()}
    vol_confirm = args.vol_confirm
    vol_confirm_window = args.vol_confirm_window

    # Binance symbol for volume confirmation
    ASSET_TO_SYMBOL: dict[str, str] = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "sol": "SOLUSDT",
        "xrp": "XRPUSDT",
    }
    binance_symbol = ASSET_TO_SYMBOL.get(asset, "BTCUSDT")

    # State dir: same directory as the trades file
    state_dir = str(Path(Config.TRADES_FILE).parent / "state")

    strategy = DeltaFlipStrategy()
    eval_params = {
        "coin": coin,
        "timeframe": timeframe,
        "gate_timeframe": gate_timeframe,
        "base_size": base_size,
        "max_size": max_size,
        "state_dir": state_dir,
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

    log(f"Strategy      : {strategy.name}")
    log(f"Asset         : {asset.upper()} / HL coin: {coin}")
    log(f"Timeframe     : {timeframe}  (fires every {window_seconds // 60} min)")
    log(f"HTF gate      : {gate_timeframe}")
    log(f"Size          : ${base_size:.2f}  max=${max_size:.2f}")
    log(f"State dir     : {state_dir}")
    log(f"Block hrs     : {sorted(block_hours) if block_hours else 'none'}")
    log(f"Vol confirm   : {'on (window=' + str(vol_confirm_window) + ')' if vol_confirm else 'off'}")
    log(f"Bankroll      : ${state.bankroll:.2f}")
    log(f"Limits        : max {Config.MAX_DAILY_BETS} bets/day, max ${Config.MAX_DAILY_LOSS} loss/day")
    log(f"Timezone      : {TIMEZONE_NAME}")
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
            log(f"Fetching HL cumulative delta for {coin} [{timeframe}]...")
            result = strategy.evaluate(_dummy_candles(), **eval_params)

            last_row = result.iloc[-1]
            raw_signal = int(last_row["signal"])
            bet_size = float(last_row["size"])

            if raw_signal == 0 or bet_size <= 0:
                log(f"No flip signal (signal={raw_signal}, size={bet_size:.2f}) — skip")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            direction = "up" if raw_signal == 1 else "down"
            log(f"Delta flip: {direction.upper()}  size=${bet_size:.2f}")

            # === VOLUME CONFIRMATION ===
            if vol_confirm:
                now_ms = int(time.time() * 1000)
                start_ms = now_ms - (vol_confirm_window + 2) * 300_000
                try:
                    vol_candles = fetch_klines(binance_symbol, "5m", start_ms, now_ms)
                    if not vol_candles.empty and len(vol_candles) >= vol_confirm_window:
                        rolling_median = vol_candles["volume"].rolling(vol_confirm_window).median().iloc[-1]
                        cur_volume = float(vol_candles["volume"].iloc[-1])
                        if cur_volume < rolling_median:
                            log(f"Vol confirm: volume {cur_volume:.0f} < median {rolling_median:.0f} — skip")
                            bet_timestamps.add(target_ts)
                            time.sleep(5)
                            continue
                        log(f"Vol confirm: volume {cur_volume:.0f} >= median {rolling_median:.0f} — pass")
                    else:
                        log("Vol confirm: not enough candles — skipping check")
                except Exception as e:
                    log(f"Vol confirm fetch error: {e} — skipping check")

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
                gate_name=f"delta_flip_{gate_timeframe}",
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
