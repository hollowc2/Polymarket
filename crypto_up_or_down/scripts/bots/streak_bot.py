#!/usr/bin/env python3
"""Polymarket BTC Streak Reversal Bot — multi-timeframe.

Supports 5m, 15m, and 1h analysis windows via --timeframe.
The bot always bets on the next 5-minute Polymarket market, but when running
at 15m or 1h it only fires once per TF-aligned window and evaluates the
signal against aggregated (resampled) outcomes at that timeframe.

Confidence (Kelly sizing) is drawn from per-timeframe REVERSAL_RATES measured
from a 2-year Binance backtest. Default trigger per timeframe:
  5m  → trigger=4  (Sharpe 3.19, 18k train trades)
  15m → trigger=6  (Sharpe 6.01,  919 train trades)
  1h  → trigger=4  (Sharpe 2.76, 1.3k train trades)
"""

import argparse
import os
import signal
import time
from datetime import datetime

import pandas as pd
from polymarket_algo.core.adapters import (
    TF_GROUP_SIZE,
    detect_streak,
    interpret_signal,
    outcomes_to_candles,
    resample_outcomes,
)
from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.core.sizing import DEFAULT_TRIGGERS
from polymarket_algo.data.binance import fetch_klines
from polymarket_algo.executor.client import PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, TradingState
from polymarket_algo.strategies.gates import TrendFilter, VolAccelGate
from polymarket_algo.strategies.session_filter import SessionFilter
from polymarket_algo.strategies.streak_adx import StreakADXStrategy
from polymarket_algo.strategies.streak_reversal import StreakReversalStrategy
from polymarket_algo.strategies.streak_rsi import StreakRSIStrategy

# Seconds per timeframe window — used to align bet windows
TF_SECONDS: dict[str, int] = {"5m": 300, "15m": 900, "1h": 3600}
# Milliseconds per candle (for Binance candle fetch window)
TF_INTERVAL_MS: dict[str, int] = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
# Binance symbol lookup
ASSET_TO_SYMBOL: dict[str, str] = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
}
# Warmup candles for EMA/RSI/ADX (covers EMA-200 with margin)
CANDLE_WARMUP = 250

# Strategy class registry
STRATEGY_MAP = {
    "streak_reversal": StreakReversalStrategy,
    "streak_rsi": StreakRSIStrategy,
    "streak_adx": StreakADXStrategy,
}

running = True


def handle_signal(sig, _frame):
    global running
    print("\n[bot] Shutting down gracefully...")
    running = False


def log(msg: str):
    ts = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def fetch_live_candles(symbol: str, interval: str, end_ms: int, warmup: int = CANDLE_WARMUP) -> pd.DataFrame | None:
    """Fetch the last *warmup* closed Binance candles for gate/indicator computation.

    Returns a DataFrame indexed by open_time (UTC), columns: open/high/low/close/volume.
    Returns None on failure so the caller can fall back to synthetic candles.
    """
    interval_ms = TF_INTERVAL_MS.get(interval, 300_000)
    start_ms = end_ms - warmup * interval_ms
    try:
        df = fetch_klines(symbol, interval, start_ms, end_ms)
        if df.empty:
            return None
        df = df.set_index("open_time").sort_index()
        # Drop any currently-open candle that snuck in
        df = df[df.index < pd.Timestamp(end_ms, unit="ms", tz="UTC")]
        return df[["open", "high", "low", "close", "volume"]] if not df.empty else None
    except Exception:
        return None  # noqa: F841 — exc intentionally swallowed; caller falls back


def candles_to_outcomes(candles: pd.DataFrame) -> list[str]:
    """Derive Polymarket-style outcome strings from real Binance candle close direction."""
    diffs = candles["close"].diff().dropna()
    return ["up" if float(d) > 0 else "down" for d in diffs]


def main():
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(
        description="Polymarket BTC Streak Reversal Bot — multi-timeframe",
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
        help="Analysis timeframe (default: %(default)s). Affects trigger default, "
        "outcome aggregation, and reversal-rate confidence table.",
    )
    parser.add_argument(
        "--trigger",
        type=int,
        metavar="N",
        help="Streak trigger length. Defaults to the best-Sharpe value for the chosen timeframe (5m→4, 15m→6, 1h→4).",
    )
    parser.add_argument("--amount", type=float, metavar="USD", help=f"Max bet amount (default: {Config.BET_AMOUNT})")
    parser.add_argument("--bankroll", type=float, metavar="USD", help="Override starting bankroll")

    # --- Strategy selection ---
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_MAP.keys()),
        default="streak_reversal",
        help="Strategy to run (default: streak_reversal)",
    )

    # --- Gate selection ---
    parser.add_argument(
        "--gate",
        choices=["none", "trend_filter", "session", "vol_accel", "trend_vol_accel"],
        default="none",
        help="Post-signal gate to apply (default: none)",
    )
    parser.add_argument(
        "--ema-period", type=int, default=100, metavar="N", help="EMA period for TrendFilter gate (default: 100)"
    )
    parser.add_argument(
        "--vol-accel-threshold",
        type=float,
        default=3.0,
        metavar="F",
        help="Vol ratio threshold for VolAccelGate (default: 3.0)",
    )
    parser.add_argument(
        "--vol-accel-boost",
        type=float,
        default=2.5,
        metavar="F",
        help="Size boost factor for VolAccelGate (default: 2.5)",
    )
    parser.add_argument(
        "--vol-accel-short-window",
        type=int,
        default=12,
        metavar="N",
        help="Short vol window (bars) for VolAccelGate (default: 12)",
    )
    parser.add_argument(
        "--vol-accel-long-window",
        type=int,
        default=240,
        metavar="N",
        help="Long (baseline) vol window (bars) for VolAccelGate (default: 240 = 20h at 5m). "
        "Also controls CANDLE_WARMUP so live and backtest windows match.",
    )
    parser.add_argument(
        "--session-hours",
        type=str,
        default="7-22",
        metavar="RANGES",
        help="UTC hour ranges for session gate. Single range: '7-22'. "
        "Multi-range: '2-3,7,9-10,17,19'. Each part is LO-HI or a single hour (default: 7-22).",
    )
    parser.add_argument(
        "--min-fill",
        type=float,
        default=0.0,
        metavar="PRICE",
        help="Minimum entry token price to accept a bet, e.g. 0.54 (default: 0.0 = off)",
    )

    # --- Strategy-specific params (override strategy defaults) ---
    parser.add_argument(
        "--rsi-period", type=int, default=None, metavar="N", help="RSI period for streak_rsi (default: 10)"
    )
    parser.add_argument(
        "--rsi-overbought",
        type=float,
        default=None,
        metavar="N",
        help="RSI overbought threshold for streak_rsi (default: 65)",
    )
    parser.add_argument(
        "--adx-period", type=int, default=None, metavar="N", help="ADX period for streak_adx (default: 14)"
    )
    parser.add_argument(
        "--adx-threshold", type=float, default=None, metavar="N", help="ADX threshold for streak_adx (default: 25)"
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
    asset = args.asset

    # Trigger: explicit flag > timeframe default > env var
    trigger = args.trigger or DEFAULT_TRIGGERS.get(timeframe, Config.STREAK_TRIGGER)
    bet_amount = args.amount or Config.BET_AMOUNT

    # --- Strategy setup ---
    strategy_name = args.strategy
    if strategy_name == "streak_rsi":
        strategy = StreakRSIStrategy(asset=asset.upper())
    elif strategy_name == "streak_adx":
        strategy = StreakADXStrategy(asset=asset.upper())
    else:
        strategy = StreakReversalStrategy()

    # Build strategy evaluate() kwargs from CLI/defaults
    eval_params: dict = {"trigger": trigger, "size": bet_amount}
    if strategy_name == "streak_rsi":
        eval_params["rsi_period"] = args.rsi_period or 10
        eval_params["rsi_overbought"] = args.rsi_overbought or 65
        eval_params["use_ci_sizing"] = False
    elif strategy_name == "streak_adx":
        eval_params["adx_period"] = args.adx_period or 14
        eval_params["adx_threshold"] = args.adx_threshold or 25
        eval_params["use_ci_sizing"] = False

    # --- Gate setup ---
    gate_type = args.gate
    trend_gate: TrendFilter | None = None
    vol_accel_gate: VolAccelGate | None = None
    session_filter: SessionFilter | None = None
    min_fill: float = args.min_fill
    if gate_type == "trend_filter":
        trend_gate = TrendFilter(ema_period=args.ema_period, mode="veto_with_trend")
    elif gate_type == "vol_accel":
        vol_accel_gate = VolAccelGate(
            short_window=args.vol_accel_short_window,
            long_window=args.vol_accel_long_window,
            threshold=args.vol_accel_threshold,
            boost_factor=args.vol_accel_boost,
        )
    elif gate_type == "trend_vol_accel":
        trend_gate = TrendFilter(ema_period=args.ema_period, mode="veto_with_trend")
        vol_accel_gate = VolAccelGate(
            short_window=args.vol_accel_short_window,
            long_window=args.vol_accel_long_window,
            threshold=args.vol_accel_threshold,
            boost_factor=args.vol_accel_boost,
        )
    elif gate_type == "session":
        allowed: list[tuple[int, int]] = []
        for part in args.session_hours.split(","):
            part = part.strip()
            if "-" in part:
                lo_str, hi_str = part.split("-", 1)
                allowed.append((int(lo_str.strip()), int(hi_str.strip())))
            elif part:
                h = int(part)
                allowed.append((h, h))
        session_filter = SessionFilter(allowed_hours=allowed or None)

    # Whether we need real Binance candles (strategies with TA indicators or gates using close prices)
    needs_real_candles = strategy_name != "streak_reversal" or gate_type in (
        "trend_filter",
        "vol_accel",
        "trend_vol_accel",
    )
    binance_symbol = ASSET_TO_SYMBOL.get(asset, "ETHUSDT")

    # Warmup must cover the longest rolling window in any active gate.
    # VolAccelGate's long_window is the binding constraint; add 22 bars of margin.
    candle_warmup = CANDLE_WARMUP
    if gate_type in ("vol_accel", "trend_vol_accel"):
        candle_warmup = max(candle_warmup, args.vol_accel_long_window + 22)

    # Init components
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
        log("LIVE trading mode - Real money!")

    _va_label = (
        f"VolAccelGate(thr={args.vol_accel_threshold}, boost={args.vol_accel_boost}, sw={args.vol_accel_short_window})"
    )
    gate_label = (
        f"TrendFilter(ema={args.ema_period}, veto_with_trend)"
        if gate_type == "trend_filter"
        else f"SessionFilter({args.session_hours} UTC)"
        if gate_type == "session"
        else _va_label
        if gate_type == "vol_accel"
        else f"TrendFilter(ema={args.ema_period}, veto_with_trend) + {_va_label}"
        if gate_type == "trend_vol_accel"
        else "none"
    )
    log(f"Strategy : {strategy.name}")
    log(f"Gate     : {gate_label}")
    log(f"Asset    : {asset.upper()}")
    log(
        f"Timeframe: {timeframe}  (fires every {window_seconds // 60} min, "
        f"trigger={trigger}, group={group_size} outcomes/bar)"
    )
    if needs_real_candles:
        log(f"Candles  : real Binance OHLCV ({binance_symbol}, {timeframe}, warmup={candle_warmup})")
    log(f"Max bet  : ${bet_amount:.2f} | Bankroll: ${state.bankroll:.2f}")
    if min_fill > 0:
        log(f"Min fill : {min_fill:.3f}")
    log(f"Limits   : max {Config.MAX_DAILY_BETS} bets/day, max ${Config.MAX_DAILY_LOSS} loss/day")
    log(f"Timezone : {TIMEZONE_NAME}")
    log("")

    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    pending: list = []

    while running:
        try:
            now = int(time.time())
            # Always align to 5m boundaries for market lookup
            current_5m = (now // 300) * 300
            seconds_into_5m = now - current_5m
            next_5m = current_5m + 300
            target_ts = next_5m  # the Polymarket market to bet on
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
                if seconds_into_5m == 0:
                    log(f"Paused: {reason}")
                time.sleep(10)
                continue

            # Already bet on this market?
            if target_ts in bet_timestamps:
                time.sleep(5)
                continue

            # === TIMEFRAME GATE — only fire on TF-aligned windows ===
            # For 5m this is always true. For 15m, fires every 3rd window.
            # For 1h, fires every 12th window.
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

            # === GET RECENT OUTCOMES ===
            # Fetch enough raw 5m outcomes to fill `trigger + 2` bars at target TF
            raw_count = (trigger + 2) * group_size
            log(f"Fetching {raw_count} outcomes (→ {trigger + 2} {timeframe} bars)...")
            outcomes_raw = client.get_recent_outcomes(count=raw_count)

            # Resample into target-timeframe bars
            outcomes = resample_outcomes(outcomes_raw, group_size)

            if len(outcomes) < trigger:
                log(f"Only {len(outcomes)} {timeframe} bars after resample, need {trigger}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            log(f"{timeframe} bars: {' -> '.join(o.upper() for o in outcomes[-trigger - 2 :])}")

            # === EVALUATE VIA STRATEGY PROTOCOL ===
            # Use real Binance OHLCV when the strategy or gate needs actual price data.
            # StreakADX needs real high/low (ADX=0 on synthetic candles → always "choppy").
            # StreakRSI needs real close prices for a meaningful RSI value.
            # TrendFilter needs real close prices for EMA computation.
            real_candles: pd.DataFrame | None = None
            if needs_real_candles:
                end_ms = current_5m * 1000  # exclude the currently-open candle
                real_candles = fetch_live_candles(binance_symbol, timeframe, end_ms, warmup=candle_warmup)
                if real_candles is None:
                    log("Binance candle fetch failed — falling back to synthetic candles")

            # For streak_reversal, always evaluate on oracle-aligned Polymarket outcomes.
            # StreakADX/RSI legitimately need Binance OHLCV for their indicators.
            if strategy_name == "streak_reversal":
                signal_candles = outcomes_to_candles(outcomes)
                outcomes_for_kelly = outcomes  # Polymarket-aligned Kelly sizing
                log("eval on synthetic (Polymarket outcomes)")
            else:
                signal_candles = real_candles if real_candles is not None else outcomes_to_candles(outcomes)
                outcomes_for_kelly = (
                    candles_to_outcomes(real_candles)
                    if real_candles is not None and not real_candles.empty
                    else outcomes
                )
            result = strategy.evaluate(signal_candles, **eval_params)

            # Capture size before gates for boost detection
            pre_gate_size = float(result.iloc[-1]["size"]) if "size" in result.columns else 0.0

            # === APPLY GATE ===
            if gate_type == "session" and session_filter is not None:
                result = session_filter.apply(result, signal_candles)
            elif gate_type in ("trend_filter", "trend_vol_accel") and trend_gate is not None:
                if real_candles is not None:
                    result = trend_gate.apply(result, real_candles)
                else:
                    log("TrendFilter skipped — no real candles available (Binance fetch failed)")

            if gate_type in ("vol_accel", "trend_vol_accel") and vol_accel_gate is not None:
                if real_candles is not None:
                    result = vol_accel_gate.apply(result, real_candles)
                else:
                    log("VolAccelGate skipped — no real candles available (Binance fetch failed)")

            # Detect gate boost and skip state
            post_gate_size = float(result.iloc[-1]["size"]) if "size" in result.columns else 0.0
            gate_boosted = post_gate_size > pre_gate_size + 0.001
            gate_skipped = gate_type in ("vol_accel", "trend_vol_accel", "trend_filter") and real_candles is None

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

            last_signal = int(result.iloc[-1]["signal"])
            entry_price = market.up_price if last_signal == 1 else market.down_price
            if entry_price <= 0:
                entry_price = 0.5

            # === MIN-FILL GATE ===
            if min_fill > 0 and last_signal != 0 and entry_price < min_fill:
                log(f"Min-fill gate: entry_price {entry_price:.3f} < {min_fill} — skip")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === INTERPRET SIGNAL (timeframe-aware confidence) ===
            decision = interpret_signal(
                result=result,
                outcomes=outcomes_for_kelly,
                bankroll=state.bankroll,
                entry_price=entry_price,
                max_bet=bet_amount,
                timeframe=timeframe,
                asset=asset.upper(),
            )

            if not decision.should_bet:
                log(f"No signal: {decision.reason}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === BANKROLL CHECK ===
            can_trade, reason = state.can_trade(bet_size=decision.size)
            if not can_trade:
                log(f"Skipping: {reason}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === PLACE BET ===
            streak_len, _ = detect_streak(outcomes_for_kelly)
            log(f"Signal: {decision.reason}")
            trade = trader.place_bet(
                market=market,
                direction=decision.direction,
                amount=decision.size,
                confidence=decision.confidence,
                streak_length=streak_len,
                gate_name=gate_type,
                gate_boosted=gate_boosted,
                gate_skipped=gate_skipped,
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
    if pending:
        state.mark_pending_as_force_exit("shutdown")
    state.save()
    log(f"State saved. Bankroll: ${state.bankroll:.2f}")
    log(f"Session: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f}")


if __name__ == "__main__":
    main()
