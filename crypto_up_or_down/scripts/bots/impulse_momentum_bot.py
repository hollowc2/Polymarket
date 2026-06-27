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
from polymarket_algo.core.discord_trades import DiscordTrades
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


@dataclass(frozen=True)
class SellExecution:
    price: float
    proceeds: float


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


def sell_execution(book: dict | None, shares: float) -> SellExecution | None:
    """Return bid-side VWAP only when every share can fill."""
    if shares <= 0:
        return None
    remaining = shares
    proceeds = 0.0
    for level in sorted((book or {}).get("bids") or [], key=lambda item: float(item["price"]), reverse=True):
        size = min(remaining, float(level["size"]))
        proceeds += size * float(level["price"])
        remaining -= size
        if remaining <= 1e-9:
            return SellExecution(price=proceeds / shares, proceeds=proceeds)
    return None


def entry_price_allowed(decision_ask: float, execution_price: float, max_drift: float) -> bool:
    return execution_price > 0 and execution_price - decision_ask <= max_drift + 1e-12


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
    entry_fee = PolymarketClient.calculate_fee_amount(shares, entry_price, trade.fee_rate_bps)
    exit_fee = PolymarketClient.calculate_fee_amount(shares, exit_price, trade.fee_rate_bps)
    pnl = gross_payout - trade.amount - entry_fee - exit_fee

    trade.shares_bought = shares
    trade.gross_payout = gross_payout
    trade.gross_profit = gross_payout - trade.amount
    trade.fee_amount = entry_fee + exit_fee
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


def trade_chart(discord: DiscordTrades, trade: Trade, exit_ms: int | None = None) -> bytes | None:
    entry_ms = trade.executed_at or trade.timestamp * 1000
    end_ms = exit_ms or int(time.time() * 1000)
    try:
        frame = fetch_klines("BTCUSDT", "1m", end_ms - 2 * 3_600_000, end_ms)
        open_seconds = frame["open_time"].map(lambda value: int(pd.Timestamp(value).timestamp()))
        window = frame.loc[(open_seconds >= trade.timestamp) & (open_seconds < trade.timestamp + 300)]
        impulse = ""
        if not window.empty:
            impulse = f" | impulse ${float(window.iloc[-1]['close']) - float(window.iloc[0]['open']):+,.0f}"
        title = f"BTC 1m spot | {trade.direction.upper()}{impulse}"
        subtitle = (
            f"entry {trade.execution_price:.3f} | bid/ask {trade.best_bid:.3f}/{trade.best_ask:.3f}"
            if exit_ms is None
            else f"entry {trade.execution_price:.3f} | exit {(trade.final_price or 0):.3f} | P&L ${trade.pnl:+.2f}"
        )
        yes_value = trade.final_price if exit_ms is not None and trade.final_price is not None else trade.best_ask
        if not yes_value:
            yes_value = trade.execution_price or trade.entry_price
        no_value = trade.opposite_price
        if no_value is None and yes_value is not None:
            no_value = max(0.0, min(1.0, 1.0 - yes_value))
        opposite = "DOWN" if trade.direction == "up" else "UP"
        return discord.chart(
            frame,
            title,
            entry_ms,
            exit_ms,
            highlight_ms=(trade.timestamp * 1000, (trade.timestamp + 300) * 1000),
            subtitle=subtitle,
            yes_label=f"YES {trade.direction.upper()}",
            yes_value=yes_value,
            no_label=f"NO {opposite}",
            no_value=no_value,
        )
    except Exception:
        return None


def notify_entry(discord: DiscordTrades, trade: Trade, impulse: float) -> None:
    discord.send(
        trade.market_slug,
        (
            f"⚡ **IMPULSE MOMENTUM ENTERED** `{trade.direction.upper()}`\n"
            f"> Market: `{trade.market_slug}`\n"
            f"> Fill: **{trade.execution_price:.3f}** | Bid/Ask: {trade.best_bid:.3f}/{trade.best_ask:.3f}\n"
            f"> Size: **${trade.amount:.2f}** | Shares: {trade.shares_bought:.4f}\n"
            f"> Impulse: ${impulse:+,.2f} | Spread: {trade.spread:.3f} | Fill: {trade.fill_pct:.0f}%\n"
            f"> Mode: {'PAPER' if trade.paper else 'LIVE'}"
        ),
        trade_chart(discord, trade),
        remember=True,
    )


def notify_exit(discord: DiscordTrades, trade: Trade) -> None:
    exit_ms = trade.settled_at or int(time.time() * 1000)
    entry_ms = trade.executed_at or trade.timestamp * 1000
    held = (exit_ms - entry_ms) / 1000
    pnl_pct = trade.pnl / trade.amount if trade.amount else 0.0
    reason = trade.force_exit_reason or trade.outcome or "resolved"
    discord.send(
        trade.market_slug,
        (
            f"{'✅' if trade.pnl >= 0 else '❌'} **IMPULSE MOMENTUM EXITED** `{trade.direction.upper()}`\n"
            f"> Market: `{trade.market_slug}`\n"
            f"> Entry: {trade.execution_price:.3f} → Exit: **{(trade.final_price or 0):.3f}**\n"
            f"> P&L: **${trade.pnl:+.2f}** ({pnl_pct:+.1%}) | Fees: ${trade.fee_amount:.2f}\n"
            f"> Held: {held:.0f}s | Reason: `{reason}`"
        ),
        trade_chart(discord, trade, exit_ms),
    )


def monitor_pending(
    state: TradingState,
    client: PolymarketClient,
    discord: DiscordTrades,
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
            notify_exit(discord, trade)
            state.save()
            continue

        token_id = None
        if market:
            token_id = market.up_token_id if trade.direction == "up" else market.down_token_id
        if not token_id:
            continue

        seconds_left = trade.timestamp + 300 - time.time()
        exit_reason = ""
        entry_price = trade.execution_price if trade.execution_price > 0 else trade.entry_price
        shares = trade.shares_bought or (trade.amount / entry_price if entry_price > 0 else 0.0)
        execution = sell_execution(client.get_orderbook(token_id), shares)
        if execution is None:
            continue
        if execution.price <= entry_price * (1.0 - stop_loss_pct):
            exit_reason = f"stop_loss_{stop_loss_pct:.0%}"
        elif seconds_left <= exit_before_sec and seconds_left > 0:
            exit_reason = f"time_exit_{exit_before_sec}s"

        if exit_reason:
            settle_paper_exit(state, trade, execution.price, exit_reason)
            log(
                f"Paper exit {trade.direction.upper()} @ {execution.price:.3f} ({exit_reason}): "
                f"${trade.pnl:+.2f} | bankroll=${state.bankroll:.2f}"
            )
            notify_exit(discord, trade)
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
    parser.add_argument("--max-selected-ask", type=float, default=0.85)
    parser.add_argument("--max-quote-age-sec", type=float, default=8.0)
    parser.add_argument("--max-entry-drift", type=float, default=0.03)
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
    discord = DiscordTrades(Config.TRADES_FILE, "impulse_momentum")
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
            monitor_pending(state, client, discord, args.exit_before_sec, args.stop_loss_pct)

            now = time.time()
            window_start = (int(now) // 300) * 300
            seconds_left = window_start + 300 - now
            earliest = args.entry_target_sec - args.entry_tolerance_sec
            latest = args.entry_target_sec + args.entry_tolerance_sec

            if window_start in traded_windows or not (earliest <= seconds_left <= latest):
                consecutive_errors = 0
                time.sleep(args.poll_sec)
                continue

            can_trade, reason = state.can_trade()
            if not can_trade:
                if reason != last_risk_pause_reason:
                    log(f"Risk pause: {reason}")
                    last_risk_pause_reason = reason
                consecutive_errors = 0
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
                consecutive_errors = 0
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
                consecutive_errors = 0
                time.sleep(args.poll_sec)
                continue

            direction = "up" if signal_value > 0 else "down"

            bet_size = portfolio_bet_size(state.bankroll, args.risk_pct, args.max_notional)
            can_trade, reason = state.can_trade(bet_size=bet_size)
            if bet_size < Config.MIN_BET or not can_trade:
                if reason != last_risk_pause_reason:
                    log(f"Risk pause: {reason}")
                    last_risk_pause_reason = reason
                traded_windows.add(window_start)
                consecutive_errors = 0
                time.sleep(args.poll_sec)
                continue

            quote_started = time.monotonic()
            up_book = book_snapshot(client.get_orderbook(market.up_token_id))
            down_book = book_snapshot(client.get_orderbook(market.down_token_id))
            quote_age = time.monotonic() - quote_started
            if not up_book or not down_book:
                raise RuntimeError("fresh CLOB order book unavailable")
            if quote_age > args.max_quote_age_sec:
                log(f"Skip stale execution quotes: {quote_age:.1f}s > {args.max_quote_age_sec:.1f}s")
                consecutive_errors = 0
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
            fresh_result = strategy.evaluate(
                frame,
                impulse_usd_min=args.impulse_min,
                threshold_price=args.threshold,
                size=1.0,
            ).iloc[-1]
            fresh_signal = int(fresh_result["signal"])
            if fresh_signal != signal_value:
                log(f"Skip changed signal: original={signal_value:+d}, fresh={fresh_signal:+d}")
                consecutive_errors = 0
                time.sleep(args.poll_sec)
                continue

            impulse = float(fresh_result["impulse_usd"])
            selected_book = up_book if direction == "up" else down_book
            if selected_book.best_ask > args.max_selected_ask:
                log(f"Skip ask {selected_book.best_ask:.3f} > {args.max_selected_ask:.3f}")
                consecutive_errors = 0
                time.sleep(args.poll_sec)
                continue
            if selected_book.spread > args.max_spread:
                log(f"Skip spread {selected_book.spread:.3f} > {args.max_spread:.3f}")
                consecutive_errors = 0
                time.sleep(args.poll_sec)
                continue
            if selected_book.top_ask_notional < args.min_ask_notional:
                log(f"Skip depth ${selected_book.top_ask_notional:.2f} < ${args.min_ask_notional:.2f}")
                consecutive_errors = 0
                time.sleep(args.poll_sec)
                continue

            token_id = market.up_token_id if direction == "up" else market.down_token_id
            execution_price, spread, slippage_pct, fill_pct, _, _ = client.get_execution_price(
                token_id, "BUY", bet_size
            )
            if fill_pct < 100.0:
                log(f"Paper FOK rejected: only {fill_pct:.1f}% of ${bet_size:.2f} fillable")
                consecutive_errors = 0
                time.sleep(args.poll_sec)
                continue
            if not entry_price_allowed(selected_book.best_ask, execution_price, args.max_entry_drift):
                log(
                    f"Skip entry drift: decision={selected_book.best_ask:.3f}, "
                    f"execution={execution_price:.3f}, limit=+{args.max_entry_drift:.3f}"
                )
                consecutive_errors = 0
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
                precomputed_execution={
                    "execution_price": execution_price,
                    "spread": spread,
                    "slippage_pct": slippage_pct,
                    "fill_pct": fill_pct,
                    "best_bid": selected_book.best_bid,
                    "best_ask": selected_book.best_ask,
                },
            )
            if trade is None:
                log("Paper order rejected")
                consecutive_errors = 0
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
            notify_entry(discord, trade, impulse)
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
