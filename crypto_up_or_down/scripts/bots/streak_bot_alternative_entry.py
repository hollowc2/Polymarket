#!/usr/bin/env python3
"""Polymarket BTC Streak Reversal Bot — alternative entry execution.

Three toggleable features layered on top of the standard streak strategy:

  1. Streak length filter  (ALT_ENTRY_USE_STREAK_FILTER / --no-streak-filter)
       Only trade streaks >= ALT_ENTRY_MIN_STREAK (default 4).

  2. Price floor filter    (ALT_ENTRY_USE_PRICE_FLOOR / --no-price-floor)
       Skip if the reversal ask > ALT_ENTRY_MAX_ENTRY_PRICE (default 0.44).
       Rationale: after N consecutive up candles the DOWN price is often
       0.40–0.45; a price above 0.44 means the crowd isn't pricing in much
       continuation momentum, reducing edge.

  3. Limit orders          (ALT_ENTRY_USE_LIMIT_ORDERS / --no-limit-orders)
       Place GTC bids at a discount instead of FOK market orders.
       Two discount modes (--discount-mode):
         fixed  — streak-length dependent table (ALT_ENTRY_DISCOUNTS, default)
         spread — fraction of current bid-ask spread (ALT_ENTRY_DISCOUNT_FRACTION)
                  e.g. fraction=0.5 places at mid-price, fraction=0.7 at 70%
                  of the way from ask toward bid.
       If the limit doesn't fill within the fill window it is cancelled and
       logged to missed_orders.json.

  4. Dynamic fill window   (ALT_ENTRY_DYNAMIC_FILL_WINDOW / --no-dynamic-fill)
       Adjusts the per-order expiry deadline based on live price movement:
         • Cancel early  if ask drifts >CANCEL_DRIFT_THRESHOLD above our limit
           (price moving away after 30 s — trade is getting stale).
         • Extend window by EXTEND_BY_SEC if ask is within EXTEND_THRESHOLD of
           our limit (almost filled — hold on a bit longer).
       Total window is capped at 2× the original fill_window.

Features apply in order; each can independently abort the trade:
    streak_filter → price_floor → limit_order

Usage:
    uv run python scripts/streak_bot_alternative_entry.py --paper
    uv run python scripts/streak_bot_alternative_entry.py --paper --discount-mode spread
    uv run python scripts/streak_bot_alternative_entry.py --paper --no-limit-orders
    uv run python scripts/streak_bot_alternative_entry.py --paper --no-streak-filter --no-price-floor
"""

import argparse
import json
import os
import signal
import time
from datetime import datetime

from polymarket_algo.core.adapters import (
    TF_GROUP_SIZE,
    detect_streak,
    interpret_signal,
    outcomes_to_candles,
    resample_outcomes,
)
from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.core.sizing import DEFAULT_TRIGGERS
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import LiveTrader, PaperTrader, TradingState
from polymarket_algo.strategies.streak_reversal import StreakReversalStrategy

# Path for logging missed limit orders (env var lets each docker service use its own file)
MISSED_ORDERS_PATH = os.getenv("MISSED_ORDERS_FILE", "missed_orders.json")

# Path for signal audit log — append-only JSONL, one entry per evaluated window
SIGNAL_LOG_PATH = os.getenv("SIGNAL_AUDIT_FILE", "signal_audit.jsonl")

# Strategy name stored in trade context (shown in monitor dashboard)
STRATEGY_NAME = "streak_reversal+limit"

# Seconds per timeframe window — used to align bet windows
TF_SECONDS: dict[str, int] = {"5m": 300, "15m": 900, "1h": 3600}

running = True


def handle_signal(sig, _frame):
    global running
    print("\n[bot] Shutting down gracefully...")
    running = False


def log(msg: str):
    ts = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _write_signal_audit(entry: dict) -> None:
    """Append one JSON line to SIGNAL_LOG_PATH. Never raises."""
    try:
        entry.setdefault("logged_at", int(time.time()))
        with open(SIGNAL_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def main():
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(
        description="Polymarket Streak Reversal Bot — alternative entry execution",
    )
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--live", action="store_true", help="Force live trading (requires PRIVATE_KEY)")
    parser.add_argument(
        "--asset",
        choices=["btc", "eth", "sol", "xrp"],
        default=os.getenv("ASSET", "btc").lower(),
        help="Asset to trade (default: btc or ASSET env var)",
    )
    parser.add_argument(
        "--timeframe",
        choices=["5m", "15m", "1h"],
        default=Config.TIMEFRAME,
        help="Analysis timeframe (default: %(default)s).",
    )
    parser.add_argument(
        "--trigger",
        type=int,
        metavar="N",
        help="Streak trigger length. Defaults to best-Sharpe per timeframe (5m→4, 15m→6, 1h→4).",
    )
    parser.add_argument("--amount", type=float, metavar="USD", help=f"Max bet amount (default: {Config.BET_AMOUNT})")
    parser.add_argument("--bankroll", type=float, metavar="USD", help="Override starting bankroll")

    # Feature toggles — mirror the env vars but let CLI override
    parser.add_argument(
        "--no-streak-filter",
        dest="streak_filter",
        action="store_false",
        default=Config.ALT_ENTRY_USE_STREAK_FILTER,
        help="Disable streak length filter",
    )
    parser.add_argument(
        "--no-price-floor",
        dest="price_floor",
        action="store_false",
        default=Config.ALT_ENTRY_USE_PRICE_FLOOR,
        help="Disable price floor filter",
    )
    parser.add_argument(
        "--no-limit-orders",
        dest="limit_orders",
        action="store_false",
        default=Config.ALT_ENTRY_USE_LIMIT_ORDERS,
        help="Disable limit orders (falls back to FOK market orders)",
    )
    parser.add_argument(
        "--min-streak",
        type=int,
        default=Config.ALT_ENTRY_MIN_STREAK,
        metavar="N",
        help=f"Minimum streak length to trade (default: {Config.ALT_ENTRY_MIN_STREAK})",
    )
    parser.add_argument(
        "--max-ask",
        type=float,
        default=Config.ALT_ENTRY_MAX_ENTRY_PRICE,
        metavar="PRICE",
        help=f"Max reversal ask price to trade (default: {Config.ALT_ENTRY_MAX_ENTRY_PRICE})",
    )
    parser.add_argument(
        "--fill-window",
        type=int,
        default=Config.ALT_ENTRY_FILL_WINDOW_SEC,
        metavar="SEC",
        help=f"Seconds to wait for limit fill before cancelling (default: {Config.ALT_ENTRY_FILL_WINDOW_SEC})",
    )
    parser.add_argument(
        "--discount-mode",
        choices=["fixed", "spread"],
        default="spread" if Config.ALT_ENTRY_USE_SPREAD_DISCOUNT else "fixed",
        help="Limit price discount mode: 'fixed' uses per-streak table, 'spread' uses fraction of bid-ask spread",
    )
    parser.add_argument(
        "--discount-fraction",
        type=float,
        default=Config.ALT_ENTRY_DISCOUNT_FRACTION,
        metavar="F",
        help=(
            f"Spread fraction for discount when --discount-mode=spread "
            f"(default: {Config.ALT_ENTRY_DISCOUNT_FRACTION}, 0.5=mid)"
        ),
    )
    parser.add_argument(
        "--no-dynamic-fill",
        dest="dynamic_fill",
        action="store_false",
        default=Config.ALT_ENTRY_DYNAMIC_FILL_WINDOW,
        help="Disable dynamic fill window (fixed expiry only)",
    )
    args = parser.parse_args()

    # Determine trading mode — default to paper
    if args.live:
        paper_mode = False
    elif args.paper:
        paper_mode = True
    else:
        paper_mode = Config.PAPER_TRADE

    timeframe = args.timeframe
    window_seconds = TF_SECONDS[timeframe]
    group_size = TF_GROUP_SIZE[timeframe]

    trigger = args.trigger or DEFAULT_TRIGGERS.get(timeframe, Config.STREAK_TRIGGER)
    bet_amount = args.amount or Config.BET_AMOUNT

    use_streak_filter = args.streak_filter
    use_price_floor = args.price_floor
    use_limit_orders = args.limit_orders
    min_streak = args.min_streak
    max_ask = args.max_ask
    fill_window = args.fill_window
    discounts = Config.ALT_ENTRY_DISCOUNTS
    use_spread_discount = args.discount_mode == "spread"
    discount_fraction = args.discount_fraction
    use_dynamic_fill = args.dynamic_fill
    cancel_drift_threshold = Config.ALT_ENTRY_CANCEL_DRIFT_THRESHOLD
    extend_threshold = Config.ALT_ENTRY_EXTEND_THRESHOLD
    extend_by_sec = Config.ALT_ENTRY_EXTEND_BY_SEC

    # Init components
    client = PolymarketClient(asset=args.asset)
    strategy = StreakReversalStrategy()
    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    if paper_mode:
        trader = PaperTrader()
        log("Paper trading mode")
    else:
        trader = LiveTrader()
        log("LIVE trading mode - Real money!")
        n_cancelled = trader.cancel_all_open_orders()
        if n_cancelled:
            log(f"[LIVE] Startup: cancelled {n_cancelled} orphaned CLOB order(s) from prior run")
        live_balance = trader.get_wallet_balance()
        if live_balance is not None:
            old = state.bankroll
            state.bankroll = live_balance
            state.save()  # persist to disk so monitor.py reads the correct value
            log(f"[LIVE] Wallet sync: ${live_balance:.2f} USDC (was ${old:.2f} simulated)")
        else:
            log("[LIVE] Wallet sync failed — keeping last saved bankroll")

    log(f"Strategy : {strategy.name} (alternative entry)")
    log(f"Asset    : {args.asset.upper()}")
    log(
        f"Timeframe: {timeframe}  (fires every {window_seconds // 60} min, "
        f"trigger={trigger}, group={group_size} outcomes/bar)"
    )
    log(f"Max bet  : ${bet_amount:.2f} | Bankroll: ${state.bankroll:.2f}")
    log(f"Limits   : max {Config.MAX_DAILY_BETS} bets/day, max ${Config.MAX_DAILY_LOSS} loss/day")
    log(f"Timezone : {TIMEZONE_NAME}")
    log("")
    log("── Alternative entry features ──────────────────────────────")
    log(f"  Streak filter  : {'ON' if use_streak_filter else 'OFF'}  (min_streak={min_streak})")
    log(f"  Price floor    : {'ON' if use_price_floor else 'OFF'}  (max_ask={max_ask:.3f})")
    if use_limit_orders:
        if use_spread_discount:
            log(
                f"  Limit orders   : ON  spread-discount (fraction={discount_fraction:.2f}, fill_window={fill_window}s)"
            )
        else:
            log(f"  Limit orders   : ON  fixed-discount (discounts={discounts}, fill_window={fill_window}s)")
        log(
            f"  Dynamic fill   : {'ON' if use_dynamic_fill else 'OFF'}  "
            f"(cancel_drift={cancel_drift_threshold:.3f}, extend_thr={extend_threshold:.3f}, "
            f"extend_by={extend_by_sec}s)"
        )
    else:
        log("  Limit orders   : OFF  (FOK market orders)")
    log("────────────────────────────────────────────────────────────")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    # Settled market orders waiting for resolution
    pending: list = []
    # Limit orders waiting for fill or expiry:
    # list of (trade, market, placed_at_unix, fill_deadline_unix)
    pending_limits: list = []

    while running:
        try:
            now = int(time.time())
            current_5m = (now // 300) * 300
            seconds_into_5m = now - current_5m
            next_5m = current_5m + 300
            target_ts = next_5m
            seconds_until_target = target_ts - now

            # === SETTLE PENDING MARKET-ORDER TRADES ===
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
                    if not paper_mode and isinstance(trader, LiveTrader):
                        # Polymarket does NOT auto-redeem for raw EOA wallets — we
                        # must call redeemPositions() on-chain to collect winnings.
                        if trade.won:
                            trader.redeem_winning_position(trade, market)
                        # Re-read on-chain balance after redemption so Kelly sizing
                        # uses the updated bankroll.
                        live_bal = trader.get_wallet_balance()
                        if live_bal is not None:
                            state.bankroll = live_bal
                            log(f"[LIVE] Wallet synced after settlement: ${live_bal:.2f} USDC")
                    state.save()

            # === POLL PENDING LIMIT ORDERS ===
            still_pending_limits = []
            for limit_trade, limit_market, placed_at, fill_deadline in pending_limits:
                seconds_elapsed = now - placed_at
                if trader.check_limit_fill(limit_trade):
                    log(
                        f"[L] Limit filled: {limit_trade.direction.upper()} "
                        f"@ {limit_trade.execution_price:.4f} "
                        f"(target {limit_trade.limit_price:.4f}) | elapsed {seconds_elapsed}s"
                    )
                    state.record_trade(limit_trade)
                    pending.append(limit_trade)
                    state.save()
                    _write_signal_audit(
                        {
                            "ts": limit_trade.timestamp,
                            "window_slug": limit_trade.market_slug,
                            "outcome": "filled",
                            "reason": "limit_filled",
                            "streak_len": limit_trade.streak_length,
                            "limit_price": limit_trade.limit_price,
                            "ask_price": limit_trade.entry_price,
                            "fill_price": limit_trade.execution_price,
                            "elapsed_sec": seconds_elapsed,
                        }
                    )
                elif now >= fill_deadline:
                    log(
                        f"[L] Limit expired after {seconds_elapsed}s "
                        f"({limit_trade.direction.upper()} @ {limit_trade.limit_price:.4f}) — missed"
                    )
                    try:
                        cancel_ok = trader.cancel_limit_bet(limit_trade, MISSED_ORDERS_PATH)
                        if not cancel_ok:
                            log(
                                f"[L] Cancel failed for {limit_trade.limit_order_id or limit_trade.order_id} "
                                f"— dropping from pending anyway"
                            )
                    except Exception as e:
                        log(f"[L] Unexpected cancel error for {limit_trade.market_slug}: {e} — dropping")
                    _write_signal_audit(
                        {
                            "ts": limit_trade.timestamp,
                            "window_slug": limit_trade.market_slug,
                            "outcome": "missed_fill",
                            "reason": "limit_expired",
                            "streak_len": limit_trade.streak_length,
                            "limit_price": limit_trade.limit_price,
                            "ask_price": limit_trade.entry_price,
                            "elapsed_sec": seconds_elapsed,
                        }
                    )
                    # Either way: do NOT append to still_pending_limits
                else:
                    remaining = fill_deadline - now
                    cancelled_early = False

                    # Dynamic fill window: adjust deadline based on live price
                    if use_dynamic_fill and use_limit_orders and limit_trade.limit_price is not None:
                        try:
                            token_id = (
                                limit_market.down_token_id
                                if limit_trade.direction == "down"
                                else limit_market.up_token_id
                            )
                            if token_id:
                                spread_data = client.get_spread(token_id)
                                if spread_data and spread_data[1] > 0:
                                    _bid, live_ask = spread_data
                                    ask_gap = live_ask - limit_trade.limit_price

                                    # Cancel early: ask has drifted away after 30s grace
                                    if ask_gap > cancel_drift_threshold and seconds_elapsed > 30:
                                        log(
                                            f"[L] Cancelling early: ask={live_ask:.4f} drifted "
                                            f"{ask_gap:.4f} above limit={limit_trade.limit_price:.4f}"
                                        )
                                        trader.cancel_limit_bet(limit_trade, MISSED_ORDERS_PATH)
                                        cancelled_early = True
                                        _write_signal_audit(
                                            {
                                                "ts": limit_trade.timestamp,
                                                "window_slug": limit_trade.market_slug,
                                                "outcome": "missed_fill",
                                                "reason": "dynamic_cancel_drift",
                                                "streak_len": limit_trade.streak_length,
                                                "limit_price": limit_trade.limit_price,
                                                "ask_price": live_ask,
                                                "ask_gap": round(ask_gap, 4),
                                                "elapsed_sec": seconds_elapsed,
                                            }
                                        )

                                    # Extend: ask is very close to our limit
                                    elif ask_gap < extend_threshold:
                                        max_deadline = placed_at + 2 * fill_window
                                        new_deadline = min(fill_deadline + extend_by_sec, max_deadline)
                                        if new_deadline > fill_deadline:
                                            log(
                                                f"[L] Extending fill window +{extend_by_sec}s: "
                                                f"ask={live_ask:.4f} within {ask_gap:.4f} of "
                                                f"limit={limit_trade.limit_price:.4f}"
                                            )
                                            fill_deadline = new_deadline
                                            remaining = fill_deadline - now
                        except Exception:
                            pass  # Never let dynamic window logic abort the poll

                    if not cancelled_early:
                        still_pending_limits.append((limit_trade, limit_market, placed_at, fill_deadline))
                        # Log progress every ~30s
                        if seconds_elapsed % 30 < 2:
                            log(
                                f"[L] Waiting for limit fill: {limit_trade.direction.upper()} "
                                f"@ {limit_trade.limit_price:.4f} | {remaining}s remaining"
                            )
            pending_limits = still_pending_limits

            # === CHECK IF WE CAN TRADE ===
            can_trade, reason = state.can_trade()
            if not can_trade:
                if seconds_into_5m == 0:
                    log(f"Paused: {reason}")
                    _write_signal_audit(
                        {
                            "ts": target_ts,
                            "window_slug": None,
                            "outcome": "skipped",
                            "reason": "daily_limit",
                            "detail": reason,
                        }
                    )
                time.sleep(10)
                continue

            # Already bet on this market?
            if target_ts in bet_timestamps:
                time.sleep(5)
                continue

            # === TIMEFRAME GATE ===
            if target_ts % window_seconds != 0:
                time.sleep(5)
                continue

            # === ENTRY TIMING ===
            if seconds_until_target > Config.ENTRY_SECONDS_BEFORE:
                if seconds_into_5m % 60 == 0:
                    log(
                        f"Next {timeframe} window in {seconds_until_target}s "
                        f"(entering at T-{Config.ENTRY_SECONDS_BEFORE}s) | "
                        f"Pending: {len(pending)} market, {len(pending_limits)} limit"
                    )
                time.sleep(1)
                continue

            # === GET RECENT OUTCOMES ===
            raw_count = (trigger + 2) * group_size
            log(f"Fetching {raw_count} outcomes (→ {trigger + 2} {timeframe} bars)...")
            outcomes_raw = client.get_recent_outcomes(count=raw_count)
            outcomes = resample_outcomes(outcomes_raw, group_size)

            if len(outcomes) < trigger:
                log(f"Only {len(outcomes)} {timeframe} bars after resample, need {trigger}")
                bet_timestamps.add(target_ts)
                _write_signal_audit(
                    {
                        "ts": target_ts,
                        "window_slug": None,
                        "outcome": "skipped",
                        "reason": "fetch_failed",
                        "detail": f"only {len(outcomes)} bars, need {trigger}",
                    }
                )
                time.sleep(5)
                continue

            log(f"{timeframe} bars: {' -> '.join(o.upper() for o in outcomes[-trigger - 2 :])}")

            # === EVALUATE VIA STRATEGY PROTOCOL ===
            candles = outcomes_to_candles(outcomes)
            result = strategy.evaluate(candles, trigger=trigger, size=bet_amount)

            # === GET TARGET MARKET ===
            market = client.get_market(target_ts)
            if not market:
                log(f"Market not found for ts={target_ts}")
                _write_signal_audit(
                    {
                        "ts": target_ts,
                        "window_slug": None,
                        "outcome": "skipped",
                        "reason": "market_not_found",
                    }
                )
                time.sleep(5)
                continue

            if not market.accepting_orders:
                log(f"Market not accepting orders: {market.slug}")
                bet_timestamps.add(target_ts)
                _write_signal_audit(
                    {
                        "ts": target_ts,
                        "window_slug": market.slug,
                        "outcome": "skipped",
                        "reason": "not_accepting",
                    }
                )
                time.sleep(5)
                continue

            last_signal = int(result.iloc[-1]["signal"])
            entry_price = market.up_price if last_signal == 1 else market.down_price
            if entry_price <= 0:
                entry_price = 0.5

            # === INTERPRET SIGNAL ===
            decision = interpret_signal(
                result=result,
                outcomes=outcomes,
                bankroll=state.bankroll,
                entry_price=entry_price,
                max_bet=bet_amount,
                timeframe=timeframe,
            )

            if not decision.should_bet:
                log(f"No signal: {decision.reason}")
                bet_timestamps.add(target_ts)
                _write_signal_audit(
                    {
                        "ts": target_ts,
                        "window_slug": market.slug,
                        "outcome": "skipped",
                        "reason": "no_signal",
                        "detail": decision.reason,
                        "ask_price": entry_price,
                    }
                )
                time.sleep(5)
                continue

            streak_len, _ = detect_streak(outcomes)
            direction = decision.direction
            size = decision.size
            confidence = decision.confidence

            # === BANKROLL CHECK ===
            can_trade, reason = state.can_trade(bet_size=size)
            if not can_trade:
                log(f"Skipping: {reason}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # Current ask for the reversal direction
            current_ask = market.down_price if direction == "down" else market.up_price

            log(f"Signal: {decision.reason} | streak={streak_len} | ask={current_ask:.4f}")

            # ── Feature 1: Streak length filter ──────────────────────────────
            if use_streak_filter and streak_len < min_streak:
                log(f"[filter] Streak {streak_len} < min {min_streak} — skipping")
                bet_timestamps.add(target_ts)
                _write_signal_audit(
                    {
                        "ts": target_ts,
                        "window_slug": market.slug,
                        "outcome": "skipped",
                        "reason": "streak_too_short",
                        "streak_len": streak_len,
                        "min_streak": min_streak,
                        "ask_price": current_ask,
                    }
                )
                time.sleep(5)
                continue

            # ── Feature 2: Price floor filter ────────────────────────────────
            if use_price_floor and current_ask > max_ask:
                log(f"[filter] Ask {current_ask:.4f} > floor {max_ask:.4f} — skipping")
                bet_timestamps.add(target_ts)
                _write_signal_audit(
                    {
                        "ts": target_ts,
                        "window_slug": market.slug,
                        "outcome": "skipped",
                        "reason": "price_floor",
                        "streak_len": streak_len,
                        "ask_price": current_ask,
                        "max_ask": max_ask,
                    }
                )
                time.sleep(5)
                continue

            # ── Feature 3: Limit orders vs market orders ──────────────────────
            if use_limit_orders:
                # Compute limit price from either spread-normalized or fixed discount
                if use_spread_discount:
                    token_id = market.down_token_id if direction == "down" else market.up_token_id
                    spread_data = client.get_spread(token_id) if token_id else None
                    if spread_data and spread_data[1] > 0:
                        best_bid, live_ask = spread_data
                        spread_width = live_ask - best_bid
                        if spread_width > 0.001:
                            discount = spread_width * discount_fraction
                            limit_price = max(0.01, round(live_ask - discount, 4))
                            log(
                                f"[limit] Spread-discount: bid={best_bid:.4f} ask={live_ask:.4f} "
                                f"spread={spread_width:.4f} → limit={limit_price:.4f} "
                                f"(fraction={discount_fraction:.2f})"
                            )
                        else:
                            # Spread too tight, fall back to fixed
                            fallback_discount = discounts.get(min(discounts.keys()), 0.03)
                            discount = discounts.get(streak_len, fallback_discount)
                            limit_price = max(0.01, round(current_ask - discount, 4))
                            log(
                                f"[limit] Spread too tight ({spread_width:.4f}), "
                                f"fixed fallback @ {limit_price:.4f} (discount={discount:.3f})"
                            )
                    else:
                        # Can't get spread, fall back to fixed discount
                        fallback_discount = discounts.get(min(discounts.keys()), 0.03)
                        discount = discounts.get(streak_len, fallback_discount)
                        limit_price = max(0.01, round(current_ask - discount, 4))
                        log(
                            f"[limit] Spread unavailable, fixed fallback @ {limit_price:.4f} "
                            f"(ask={current_ask:.4f}, discount={discount:.3f})"
                        )
                else:
                    # Fixed streak-length-dependent discount
                    fallback_discount = discounts.get(min(discounts.keys()), 0.03)
                    discount = discounts.get(streak_len, fallback_discount)
                    limit_price = max(0.01, round(current_ask - discount, 4))
                    log(
                        f"[limit] Fixed-discount: ask={current_ask:.4f} discount={discount:.3f} "
                        f"→ limit={limit_price:.4f} (streak={streak_len})"
                    )

                trade = trader.place_limit_bet(
                    market=market,
                    direction=direction,
                    amount=size,
                    limit_price=limit_price,
                    confidence=confidence,
                    streak_length=streak_len,
                )

                if trade is None:
                    log("Limit order rejected")
                    bet_timestamps.add(target_ts)
                    _write_signal_audit(
                        {
                            "ts": target_ts,
                            "window_slug": market.slug,
                            "outcome": "skipped",
                            "reason": "circuit_open",
                            "streak_len": streak_len,
                            "ask_price": current_ask,
                        }
                    )
                    continue

                trade.strategy = STRATEGY_NAME

                # Track in pending_limits — NOT yet recorded in state
                placed_at_ts = int(time.time())
                pending_limits.append((trade, market, placed_at_ts, placed_at_ts + fill_window))
                bet_timestamps.add(target_ts)
                _write_signal_audit(
                    {
                        "ts": target_ts,
                        "window_slug": market.slug,
                        "outcome": "placed",
                        "reason": "limit_placed",
                        "streak_len": streak_len,
                        "ask_price": current_ask,
                        "limit_price": limit_price,
                        "direction": direction,
                        "amount": size,
                    }
                )

                log(
                    f"Daily: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f} "
                    f"| Bankroll: ${state.bankroll:.2f} | Limits pending: {len(pending_limits)}"
                )

            else:
                # Fall back to standard FOK market order
                trade = trader.place_bet(
                    market=market,
                    direction=direction,
                    amount=size,
                    confidence=confidence,
                    streak_length=streak_len,
                    strategy=STRATEGY_NAME,
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
    # Cancel any open limit orders
    for limit_trade, _limit_market, _placed_at, _fill_deadline in pending_limits:
        log(f"Cancelling open limit: {limit_trade.direction.upper()} @ {limit_trade.limit_price:.4f}")
        trader.cancel_limit_bet(limit_trade, MISSED_ORDERS_PATH)

    if pending:
        state.mark_pending_as_force_exit("shutdown")
    state.save()
    log(f"State saved. Bankroll: ${state.bankroll:.2f}")
    log(f"Session: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f}")
    log(f"Missed limits logged to: {MISSED_ORDERS_PATH}")


if __name__ == "__main__":
    main()
