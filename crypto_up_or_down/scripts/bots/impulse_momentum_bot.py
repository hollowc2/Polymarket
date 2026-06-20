#!/usr/bin/env python3
"""Paper bot for BTC 5-minute impulse momentum into the market close."""

from __future__ import annotations

import argparse
import signal
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from polymarket_algo.core.config import LOCAL_TZ, Config
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.executor.client import Market, PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, Trade, TradingState
from polymarket_algo.strategies.impulse_momentum import ImpulseMomentumStrategy

running = True


@dataclass(frozen=True)
class BookSnapshot:
    best_bid: float
    best_ask: float
    spread: float
    top_ask_notional: float


def handle_signal(_sig, _frame):
    global running
    running = False


def log(message: str) -> None:
    timestamp = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def book_snapshot(book: dict) -> BookSnapshot | None:
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not asks:
        return None
    best_ask_level = min(asks, key=lambda level: float(level["price"]))
    best_bid = 0.0
    if bids:
        best_bid_level = max(bids, key=lambda level: float(level["price"]))
        best_bid = float(best_bid_level["price"])
    best_ask = float(best_ask_level["price"])
    ask_size = float(best_ask_level["size"])
    return BookSnapshot(
        best_bid=best_bid,
        best_ask=best_ask,
        spread=max(0.0, best_ask - best_bid),
        top_ask_notional=best_ask * ask_size,
    )


def portfolio_bet_size(bankroll: float, risk_pct: float, max_notional: float) -> float:
    """Size as a percentage of AUM; max_notional <= 0 disables the dollar cap."""
    target = bankroll * risk_pct / 100.0
    if max_notional > 0:
        target = min(target, max_notional)
    return round(min(target, bankroll), 2)


def settle_paper_exit(
    state: TradingState,
    trade: Trade,
    exit_price: float,
    reason: str,
) -> None:
    """Close a paper position at an executable bid and realize cash-flow P&L."""
    entry_price = trade.execution_price if trade.execution_price > 0 else trade.entry_price
    shares = trade.amount / entry_price if entry_price > 0 else 0.0
    gross_payout = shares * exit_price
    exit_fee_pct = PolymarketClient.calculate_fee(exit_price, trade.fee_rate_bps)
    exit_fee = gross_payout * exit_fee_pct
    pnl = gross_payout - exit_fee - trade.amount

    trade.shares_bought = shares
    trade.gross_payout = gross_payout
    trade.gross_profit = gross_payout - trade.amount
    trade.fee_amount += exit_fee
    trade.net_profit = pnl
    trade.pnl = pnl
    trade.won = pnl > 0
    trade.outcome = "paper_exit"
    trade.final_price = exit_price
    trade.price_at_close = exit_price
    trade.settled_at = int(time.time() * 1000)
    trade.settlement_status = "settled"
    trade.force_exit_reason = reason

    state.daily_pnl += pnl
    state.bankroll += pnl


def current_impulse(symbol: str, window_start: int) -> tuple[float, float]:
    now_ms = int(time.time() * 1000)
    candles = fetch_klines(symbol, "5m", window_start * 1000, now_ms)
    if candles.empty:
        raise RuntimeError("no current BTC candle")
    current = candles.iloc[-1]
    open_time = pd.Timestamp(current["open_time"]).timestamp()
    if int(open_time) != window_start:
        raise RuntimeError(f"stale BTC candle: expected {window_start}, got {int(open_time)}")
    return float(current["open"]), float(current["close"])


def monitor_pending(
    state: TradingState,
    client: PolymarketClient,
    exit_before_sec: int,
    stop_loss_pct: float,
) -> None:
    for trade in [item for item in state.trades if item.settlement_status == "pending"]:
        market = client.get_market(trade.timestamp, use_cache=False)
        if market and market.closed and market.outcome:
            state.settle_trade(trade, market.outcome, market)
            log(
                f"Resolved {trade.direction.upper()} on {market.slug}: "
                f"${trade.pnl:+.2f} | bankroll=${state.bankroll:.2f}"
            )
            state.save()
            continue

        token_id = None
        if market:
            token_id = market.up_token_id if trade.direction == "up" else market.down_token_id
        if not token_id:
            continue

        seconds_left = trade.timestamp + 300 - time.time()
        exit_reason = ""
        bid = client.get_price(token_id, "SELL")
        if bid is None:
            continue
        entry_price = trade.execution_price if trade.execution_price > 0 else trade.entry_price
        if bid <= entry_price * (1.0 - stop_loss_pct):
            exit_reason = f"stop_loss_{stop_loss_pct:.0%}"
        elif seconds_left <= exit_before_sec and seconds_left > 0:
            exit_reason = f"time_exit_{exit_before_sec}s"

        if exit_reason:
            settle_paper_exit(state, trade, bid, exit_reason)
            log(
                f"Paper exit {trade.direction.upper()} @ {bid:.3f} ({exit_reason}): "
                f"${trade.pnl:+.2f} | bankroll=${state.bankroll:.2f}"
            )
            state.save()


def place_optional_hedge(
    *,
    trader: PaperTrader,
    state: TradingState,
    market: Market,
    main_direction: str,
    selected_ask: float,
    trigger_price: float,
    hedge_amount: float,
) -> None:
    if selected_ask < trigger_price or hedge_amount < Config.MIN_BET:
        return
    hedge_direction = "down" if main_direction == "up" else "up"
    can_trade, _ = state.can_trade(bet_size=hedge_amount)
    if not can_trade:
        return
    hedge = trader.place_bet(
        market=market,
        direction=hedge_direction,
        amount=hedge_amount,
        confidence=1.0 - selected_ask,
        streak_length=0,
        strategy="impulse_momentum_hedge",
        gate_name="extreme_skew_hedge",
    )
    if hedge:
        state.record_trade(hedge)
        log(f"Optional hedge: {hedge_direction.upper()} ${hedge.amount:.2f}")


def main() -> None:
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(description="BTC 5m impulse-momentum paper bot")
    parser.add_argument("--paper", action="store_true", help="Paper mode (the only supported mode)")
    parser.add_argument(
        "--max-notional",
        type=float,
        default=0.0,
        help="Optional dollar cap; 0 disables the cap",
    )
    parser.add_argument("--risk-pct", type=float, default=10.0, help="Percent of current bankroll per trade")
    parser.add_argument("--impulse-min", type=float, default=70.0)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--entry-target-sec", type=int, default=120)
    parser.add_argument("--entry-tolerance-sec", type=int, default=30)
    parser.add_argument("--max-spread", type=float, default=0.03)
    parser.add_argument("--min-ask-notional", type=float, default=30.0)
    parser.add_argument("--max-quote-age-sec", type=float, default=8.0)
    parser.add_argument("--stop-loss-pct", type=float, default=0.25)
    parser.add_argument("--exit-before-sec", type=int, default=20)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--max-api-errors", type=int, default=3)
    parser.add_argument("--error-cooldown-sec", type=float, default=60.0)
    parser.add_argument("--hedge", action="store_true")
    parser.add_argument("--hedge-trigger", type=float, default=0.95)
    parser.add_argument("--hedge-amount", type=float, default=1.0)
    parser.add_argument("--bankroll", type=float)
    args = parser.parse_args()

    client = PolymarketClient(asset="btc", use_cache=False)
    trader = PaperTrader()
    strategy = ImpulseMomentumStrategy()
    state = TradingState.load()
    if args.bankroll is not None:
        state.bankroll = args.bankroll
    state.save()

    traded_windows = {trade.timestamp for trade in state.trades}
    consecutive_errors = 0
    last_risk_pause_reason: str | None = None

    log("PAPER ONLY — live order placement is not enabled")
    log(
        f"BTC/5m impulse>=${args.impulse_min:.0f}, CLOB threshold={args.threshold:.2f}, "
        f"entry={args.entry_target_sec}±{args.entry_tolerance_sec}s left"
    )
    cap_label = f"${args.max_notional:.2f}" if args.max_notional > 0 else "none"
    log(f"Portfolio sizing: {args.risk_pct:.2f}% of AUM per trade, dollar cap={cap_label}")

    while running:
        try:
            monitor_pending(state, client, args.exit_before_sec, args.stop_loss_pct)

            now = time.time()
            window_start = (int(now) // 300) * 300
            seconds_left = window_start + 300 - now
            earliest = args.entry_target_sec - args.entry_tolerance_sec
            latest = args.entry_target_sec + args.entry_tolerance_sec

            if window_start in traded_windows or not (earliest <= seconds_left <= latest):
                time.sleep(args.poll_sec)
                continue

            can_trade, reason = state.can_trade()
            if not can_trade:
                if reason != last_risk_pause_reason:
                    log(f"Risk pause: {reason}")
                    last_risk_pause_reason = reason
                time.sleep(args.poll_sec)
                continue
            last_risk_pause_reason = None

            market = client.get_market(window_start, use_cache=False)
            if not market or market.closed or not market.accepting_orders:
                raise RuntimeError("active BTC 5m market unavailable")
            if not market.up_token_id or not market.down_token_id:
                raise RuntimeError("market token IDs unavailable")

            quote_started = time.monotonic()
            up_book = book_snapshot(client.get_orderbook(market.up_token_id))
            down_book = book_snapshot(client.get_orderbook(market.down_token_id))
            quote_age = time.monotonic() - quote_started
            if not up_book or not down_book:
                raise RuntimeError("CLOB order book unavailable")
            if quote_age > args.max_quote_age_sec:
                log(f"Skip stale quote cycle: {quote_age:.1f}s > {args.max_quote_age_sec:.1f}s")
                time.sleep(args.poll_sec)
                continue

            interval_open, spot_price = current_impulse("BTCUSDT", window_start)
            frame = pd.DataFrame(
                [
                    {
                        "open": interval_open,
                        "close": spot_price,
                        "up_ask": up_book.best_ask,
                        "down_ask": down_book.best_ask,
                    }
                ]
            )
            result = strategy.evaluate(
                frame,
                impulse_usd_min=args.impulse_min,
                threshold_price=args.threshold,
                size=1.0,
            ).iloc[-1]
            signal_value = int(result["signal"])
            impulse = float(result["impulse_usd"])
            if signal_value == 0:
                log(
                    f"No entry: impulse=${impulse:+.2f}, "
                    f"UP ask={up_book.best_ask:.3f}, DOWN ask={down_book.best_ask:.3f}"
                )
                time.sleep(args.poll_sec)
                continue

            direction = "up" if signal_value > 0 else "down"
            selected_book = up_book if direction == "up" else down_book
            if selected_book.spread > args.max_spread:
                log(f"Skip spread {selected_book.spread:.3f} > {args.max_spread:.3f}")
                time.sleep(args.poll_sec)
                continue
            if selected_book.top_ask_notional < args.min_ask_notional:
                log(f"Skip depth ${selected_book.top_ask_notional:.2f} < ${args.min_ask_notional:.2f}")
                time.sleep(args.poll_sec)
                continue

            bet_size = portfolio_bet_size(state.bankroll, args.risk_pct, args.max_notional)
            can_trade, reason = state.can_trade(bet_size=bet_size)
            if bet_size < Config.MIN_BET or not can_trade:
                if reason != last_risk_pause_reason:
                    log(f"Risk pause: {reason}")
                    last_risk_pause_reason = reason
                traded_windows.add(window_start)
                time.sleep(args.poll_sec)
                continue

            # PaperTrader records market-side prices as signal context. Replace
            # Gamma's slower outcomePrices snapshot with the CLOB asks used by
            # this decision so entry context and executable pricing agree.
            market.up_price = up_book.best_ask
            market.down_price = down_book.best_ask
            trade = trader.place_bet(
                market=market,
                direction=direction,
                amount=bet_size,
                confidence=selected_book.best_ask,
                streak_length=0,
                strategy="impulse_momentum",
                gate_name="btc_impulse_clob_skew",
            )
            if trade is None:
                log("Paper order rejected")
                time.sleep(args.poll_sec)
                continue

            state.record_trade(trade)
            place_optional_hedge(
                trader=trader,
                state=state,
                market=market,
                main_direction=direction,
                selected_ask=selected_book.best_ask,
                trigger_price=args.hedge_trigger,
                hedge_amount=args.hedge_amount if args.hedge else 0.0,
            )
            traded_windows.add(window_start)
            state.save()
            log(
                f"Entered {direction.upper()} ${trade.amount:.2f}: impulse=${impulse:+.2f}, "
                f"ask={selected_book.best_ask:.3f}, spread={selected_book.spread:.3f}"
            )
            consecutive_errors = 0
            time.sleep(args.poll_sec)

        except KeyboardInterrupt:
            break
        except Exception as exc:
            consecutive_errors += 1
            log(f"Cycle error {consecutive_errors}/{args.max_api_errors}: {exc}")
            if consecutive_errors >= args.max_api_errors:
                log(f"API kill-switch cooldown: {args.error_cooldown_sec:.0f}s")
                time.sleep(args.error_cooldown_sec)
                consecutive_errors = 0
            else:
                time.sleep(args.poll_sec)

    state.save()
    log("Stopped cleanly; pending paper positions remain recoverable on restart")


if __name__ == "__main__":
    main()
