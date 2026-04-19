#!/usr/bin/env python3
"""Polymarket BTC 5-Min 3-Bar Momentum Bot.

Uses ThreeBarMoMoStrategy: N consecutive candles in the same direction with
strictly increasing volume → bet WITH the momentum.  Position size scales
with the volume-expansion ratio (capped at size_cap × base_size).

Live data is fetched from Binance OHLCV (real open/high/low/close/volume),
not from outcomes_to_candles, so the volume condition can be evaluated.
"""

import argparse
import signal
import time
from datetime import datetime

from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.indicators.hl_orderflow import hl_orderflow_signal
from polymarket_algo.indicators.regime import regime_ok
from polymarket_algo.strategies.three_bar_momo import ThreeBarMoMoStrategy

running = True

# Seconds before the new window to re-confirm the signal (after initial check)
_CONFIRM_SECONDS_BEFORE = 5


def handle_signal(sig, _frame):
    global running
    print("\n[bot] Shutting down gracefully...")
    running = False


def log(msg: str):
    ts = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _check_hl_gate(signal_dir: int, hl_coin: str) -> tuple[bool, str]:
    """Return (vetoed, reason). Veto when both 5m+15m strongly oppose signal."""
    sig_5m = hl_orderflow_signal(hl_coin, "5m")
    sig_15m = hl_orderflow_signal(hl_coin, "15m")
    if signal_dir == 1 and sig_5m == "SELL" and sig_15m == "SELL":
        return True, f"HL gate veto UP: {hl_coin} 5m={sig_5m} 15m={sig_15m}"
    if signal_dir == -1 and sig_5m == "BUY" and sig_15m == "BUY":
        return True, f"HL gate veto DOWN: {hl_coin} 5m={sig_5m} 15m={sig_15m}"
    return False, f"{sig_5m}/{sig_15m}"


def _get_ask_depth(client: PolymarketClient, token_id: str) -> float:
    """Return best ask-level USD depth (price × size). 0.0 on failure."""
    try:
        book = client.get_orderbook(token_id)
        if not book:
            return 0.0
        asks = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
        if not asks:
            return 0.0
        return float(asks[0]["price"]) * float(asks[0]["size"])
    except Exception:
        return 0.0


def main():
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(
        description="Polymarket BTC 5-Min 3-Bar Momentum Bot",
    )
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--live", action="store_true", help="Force live trading (requires PRIVATE_KEY)")
    parser.add_argument(
        "--bars",
        type=int,
        default=3,
        metavar="N",
        help="Consecutive qualifying bars required (default: 3)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        metavar="USD",
        help=f"Base bet amount in USD (default: {Config.BET_AMOUNT})",
    )
    parser.add_argument(
        "--size-cap",
        type=float,
        default=2.0,
        metavar="X",
        help="Max volume-expansion multiplier for bet size (default: 2.0)",
    )
    parser.add_argument(
        "--min-body-pct",
        type=float,
        default=0.0,
        metavar="F",
        help="Min candle body as fraction of close, e.g. 0.001 (default: 0.0 = off)",
    )
    parser.add_argument("--bankroll", type=float, metavar="USD", help="Override starting bankroll")
    parser.add_argument(
        "--block-hours",
        type=str,
        default="9,16,17,20",
        metavar="H,H,...",
        help="Comma-separated UTC hours to skip (default: 9,16,17,20)",
    )
    parser.add_argument(
        "--hl-gate",
        action="store_true",
        help="Veto signals when HL 5m+15m both oppose the momentum direction",
    )
    parser.add_argument(
        "--hl-coin",
        default="BTC",
        metavar="COIN",
        help="HL coin for gate check: BTC, ETH, SOL, XRP (default: BTC)",
    )
    parser.add_argument(
        "--min-market-vol",
        type=float,
        default=Config.MIN_MARKET_VOL,
        metavar="USD",
        help=f"Skip markets with total volume below this USD threshold (default: {Config.MIN_MARKET_VOL})",
    )
    parser.add_argument(
        "--min-ask-depth-mult",
        type=float,
        default=Config.MIN_ASK_DEPTH_MULT,
        metavar="X",
        help=f"Require ask-side depth >= bet_size × X before placing (default: {Config.MIN_ASK_DEPTH_MULT})",
    )
    parser.add_argument(
        "--regime-gate",
        action="store_true",
        help="Skip signals when BTC 1h ATR percentile rank is below --regime-atr-floor (ranging regime)",
    )
    parser.add_argument(
        "--regime-atr-floor",
        type=float,
        default=25.0,
        metavar="PCT",
        help="ATR percentile rank floor for regime gate (default: 25.0 = bottom quartile = ranging)",
    )
    args = parser.parse_args()

    # Determine trading mode — default to paper
    if args.live:
        paper_mode = False
    elif args.paper:
        paper_mode = True
    else:
        paper_mode = Config.PAPER_TRADE

    bars = args.bars
    bet_amount = args.amount or Config.BET_AMOUNT
    size_cap = args.size_cap
    min_body_pct = args.min_body_pct
    hl_gate = args.hl_gate
    hl_coin = args.hl_coin.upper()
    block_hours: set[int] = {int(h.strip()) for h in args.block_hours.split(",") if h.strip()}
    min_market_vol = args.min_market_vol
    min_ask_depth_mult = args.min_ask_depth_mult
    use_regime_gate = args.regime_gate
    regime_atr_floor = args.regime_atr_floor

    # Init components
    client = PolymarketClient()
    strategy = ThreeBarMoMoStrategy()
    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    if paper_mode:
        trader = PaperTrader()
        log("Paper trading mode")
    else:
        from polymarket_algo.executor.trader import LiveTrader

        trader = LiveTrader()
        log("LIVE trading mode - Real money!")

    gate_info = f", hl_gate={hl_coin}" if hl_gate else ""
    regime_info = f", regime_gate(atr≥p{regime_atr_floor:.0f})" if use_regime_gate else ""
    log(
        f"Strategy: {strategy.name} "
        f"(bars={bars}, base_bet=${bet_amount:.2f}, size_cap={size_cap}x, "
        f"min_body_pct={min_body_pct}{gate_info}{regime_info})"
    )
    log(f"Block hours (UTC): {sorted(block_hours) if block_hours else 'none'}")
    log(f"Min market vol: ${min_market_vol:.0f} | Min ask depth: {min_ask_depth_mult}× bet")
    log(f"Bankroll: ${state.bankroll:.2f}")
    log(f"Limits: max {Config.MAX_DAILY_BETS} bets/day, max ${Config.MAX_DAILY_LOSS} loss/day, max {Config.MAX_CONSEC_LOSSES} consec losses")
    log(f"Timezone: {TIMEZONE_NAME}")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    pending: list = []

    while running:
        try:
            now = int(time.time())
            current_window = (now // 300) * 300
            seconds_into_window = now - current_window
            next_window = current_window + 300
            target_ts = next_window
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

            # === CHECK IF WE CAN TRADE ===
            can_trade, reason = state.can_trade()
            if not can_trade:
                if seconds_into_window == 0:
                    log(f"Paused: {reason}")
                time.sleep(10)
                continue

            # Already bet on this market?
            if target_ts in bet_timestamps:
                time.sleep(5)
                continue

            # === ENTRY TIMING ===
            if seconds_until_target > Config.ENTRY_SECONDS_BEFORE:
                if seconds_into_window % 60 == 0:
                    log(
                        f"Next window in {seconds_until_target}s "
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

            # === REGIME GATE (1h ATR percentile) ===
            if use_regime_gate and not regime_ok(atr_pct_floor=regime_atr_floor):
                log(f"Regime gate: BTC ATR below p{regime_atr_floor:.0f} (ranging) — skip")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === FETCH BINANCE CANDLES (initial) ===
            log("Fetching Binance candles...")
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - (bars + 3) * 5 * 60 * 1000
            try:
                candles = fetch_klines("BTCUSDT", "5m", start_ms, now_ms)
            except Exception as e:
                log(f"Binance fetch error: {e}")
                time.sleep(10)
                continue

            if candles.empty or len(candles) < bars:
                log(f"Not enough candles: {len(candles)} (need {bars})")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            candles = candles.tail(bars + 2)

            # === EVALUATE STRATEGY (without HL gate — gate handled separately for metadata) ===
            result = strategy.evaluate(
                candles,
                bars=bars,
                size=bet_amount,
                size_cap=size_cap,
                min_body_pct=min_body_pct,
                hl_gate=False,  # gate checked below so we can record metadata
            )

            initial_signal = int(result.iloc[-1]["signal"])
            last_size = float(result.iloc[-1]["size"])

            if initial_signal == 0:
                log("No momentum signal on last bar")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            initial_direction = "up" if initial_signal == 1 else "down"

            # === HL GATE CHECK (with metadata recording) ===
            hl_gate_vetoed = False
            hl_gate_label = ""
            if hl_gate:
                hl_gate_vetoed, hl_gate_label = _check_hl_gate(initial_signal, hl_coin)
                if hl_gate_vetoed:
                    log(f"{hl_gate_label} — skip")
                    bet_timestamps.add(target_ts)
                    time.sleep(5)
                    continue

            # === SIGNAL RE-CONFIRMATION AT T-5s ===
            wait_until = target_ts - _CONFIRM_SECONDS_BEFORE
            seconds_to_wait = wait_until - int(time.time())
            if seconds_to_wait > 0:
                log(
                    f"Signal {initial_direction.upper()} confirmed — re-checking at T-{_CONFIRM_SECONDS_BEFORE}s "
                    f"(waiting {seconds_to_wait}s)"
                )
                time.sleep(max(0, seconds_to_wait))

            now_ms2 = int(time.time() * 1000)
            start_ms2 = now_ms2 - (bars + 3) * 5 * 60 * 1000
            try:
                candles2 = fetch_klines("BTCUSDT", "5m", start_ms2, now_ms2)
                candles2 = candles2.tail(bars + 2)
                result2 = strategy.evaluate(
                    candles2,
                    bars=bars,
                    size=bet_amount,
                    size_cap=size_cap,
                    min_body_pct=min_body_pct,
                    hl_gate=False,
                )
                final_signal = int(result2.iloc[-1]["signal"])
                last_size = float(result2.iloc[-1]["size"])
            except Exception as e:
                log(f"Re-confirmation fetch error: {e} — using initial signal")
                final_signal = initial_signal

            if final_signal != initial_signal:
                log(
                    f"Signal reversed at T-{_CONFIRM_SECONDS_BEFORE}s "
                    f"({initial_direction.upper()} → {'NONE' if final_signal == 0 else ('UP' if final_signal == 1 else 'DOWN')}) — skip"
                )
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            direction = initial_direction

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

            # === VOLUME FLOOR ===
            market_vol = getattr(market, "volume", 0.0) or 0.0
            if market_vol < min_market_vol:
                log(f"Market volume too low: ${market_vol:.0f} < ${min_market_vol:.0f} — skip")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # Clamp size: volume-scaled (from strategy) but never exceeds bet_amount
            # and never more than 10% of bankroll
            bet_size = max(1.0, min(last_size, bet_amount, state.bankroll * 0.1))

            log(f"Signal: {direction.upper()} | vol=${market_vol:.0f} | vol-scaled size=${last_size:.2f} -> capped=${bet_size:.2f}")

            # === MIN ASK-DEPTH CHECK ===
            if min_ask_depth_mult > 0:
                token_id = market.up_token_id if direction == "up" else market.down_token_id
                if token_id:
                    ask_depth = _get_ask_depth(client, token_id)
                    required = bet_size * min_ask_depth_mult
                    if ask_depth < required:
                        log(f"Insufficient ask depth: ${ask_depth:.2f} < ${required:.2f} required — skip")
                        bet_timestamps.add(target_ts)
                        time.sleep(5)
                        continue

            # === BANKROLL CHECK ===
            can_trade, reason = state.can_trade(bet_size=bet_size)
            if not can_trade:
                log(f"Skipping: {reason}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === PLACE BET ===
            trade = trader.place_bet(
                market=market,
                direction=direction,
                amount=bet_size,
                confidence=0.55,  # fixed momentum confidence
                streak_length=bars,
            )

            if trade is None:
                log("Order rejected")
                bet_timestamps.add(target_ts)
                continue

            # === RECORD GATE METADATA ===
            if hl_gate:
                trade.gate_name = "hl_orderflow"
                trade.gate_boosted = False
                trade.gate_skipped = False

            state.record_trade(trade)
            bet_timestamps.add(target_ts)
            pending.append(trade)
            state.save()

            log(
                f"Daily: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f} "
                f"| Bankroll: ${state.bankroll:.2f} | Pending: {len(pending)}"
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
