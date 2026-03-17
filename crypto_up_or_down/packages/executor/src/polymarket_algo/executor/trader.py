"""Trading execution — paper and live modes."""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from polymarket_algo.core.config import LOCAL_TZ, TIMEZONE_NAME, Config
from polymarket_algo.executor.client import Market, PolymarketClient
from polymarket_algo.executor.resilience import (
    CircuitBreaker,
    ErrorCategory,
    RateLimiter,
    categorize_error,
    with_retry,
)


@dataclass
class Trade:
    """Record of a trade (paper or live) with full history."""

    # === CORE FIELDS ===
    timestamp: int  # market timestamp (unix seconds)
    market_slug: str  # e.g., "btc-updown-5m-1771051500"
    direction: str  # "up" or "down" - your bet direction
    amount: float  # your bet size in USD (after any partial fill)
    entry_price: float  # displayed market price when you decided to bet
    streak_length: int  # for streak strategy
    confidence: float  # signal confidence (0-1)
    paper: bool  # True = simulation, False = live trade

    # === RESOLUTION FIELDS ===
    outcome: str | None = None  # "up" or "down" after market closes
    pnl: float = 0.0  # net profit/loss after fees
    order_id: str | None = None  # order ID from exchange (live only)
    settled_at: int | None = None  # when trade was settled (unix ms)
    won: bool | None = None  # True if direction == outcome

    # === SETTLEMENT BREAKDOWN ===
    shares_bought: float = 0.0  # number of shares purchased
    gross_payout: float = 0.0  # total payout before fees ($1/share if won)
    gross_profit: float = 0.0  # gross_payout - amount (before fees)
    fee_amount: float = 0.0  # actual fee deducted in USD
    net_profit: float = 0.0  # gross_profit - fee_amount

    # === COPYTRADE FIELDS ===
    copied_from: str | None = None  # trader wallet address
    trader_name: str | None = None  # trader pseudonym
    trader_direction: str | None = None  # what trader bet on
    trader_amount: float | None = None  # how much trader bet (USD)
    trader_price: float | None = None  # price trader got
    trader_timestamp: int | None = None  # when trader placed bet (unix ms)
    executed_at: int | None = None  # when you placed your bet (unix ms)
    copy_delay_ms: int | None = None  # delay between trader and your bet
    market_price_at_copy: float | None = None  # market price when you copied

    # === STRATEGY ===
    strategy: str = "streak"  # "streak" or "copytrade"

    # === REALISTIC SIMULATION FIELDS ===
    fee_rate_bps: int = 0  # base fee in basis points (e.g., 1000)
    fee_pct: float = 0.0  # actual fee percentage at execution price
    spread: float = 0.0  # bid-ask spread at entry (in price units)
    slippage_pct: float = 0.0  # slippage from walking the book (%)
    execution_price: float = 0.0  # actual fill price after slippage
    fill_pct: float = 100.0  # percentage of order filled
    delay_impact_pct: float = 0.0  # price impact from copy delay (%)
    requested_amount: float = 0.0  # original requested amount before partial fill

    # === PRICE MOVEMENT ===
    price_at_signal: float = 0.0  # price when signal was generated
    price_at_execution: float = 0.0  # price when order was submitted
    price_movement_pct: float = 0.0  # % change from signal to execution

    # === MARKET CONTEXT ===
    market_volume: float = 0.0  # market volume at time of trade
    best_bid: float = 0.0  # best bid price at execution
    best_ask: float = 0.0  # best ask price at execution

    # === ORDER STATUS (live trading) ===
    order_status: str = "pending"  # pending, submitted, filled, cancelled, failed

    # === UNREALIZED P&L (for pending trades) ===
    current_price: float | None = None  # current market price for our direction
    unrealized_pnl: float | None = None  # estimated PnL based on current price
    implied_outcome: str | None = None  # "up" or "down" based on which side > 50%

    # === PATTERN ANALYSIS FIELDS ===
    # Time-based patterns
    hour_utc: int | None = None  # Hour of day (0-23) in UTC
    minute_of_hour: int | None = None  # Minute within the hour (0-59)
    day_of_week: int | None = None  # Day of week (0=Monday, 6=Sunday)
    seconds_into_window: int | None = None  # How many seconds after window opened we entered

    # Session tracking
    session_trade_number: int | None = None  # Which trade # in this session
    session_wins_before: int | None = None  # Session wins before this trade
    session_losses_before: int | None = None  # Session losses before this trade
    session_pnl_before: float | None = None  # Session PnL before this trade
    bankroll_before: float | None = None  # Bankroll before this trade

    # Streak tracking
    consecutive_wins: int = 0  # How many wins in a row before this trade
    consecutive_losses: int = 0  # How many losses in a row before this trade

    # Market context
    opposite_price: float | None = None  # Price of the opposite outcome
    price_ratio: float | None = None  # our_price / opposite_price
    market_bias: str | None = None  # "bullish" (up>down), "bearish", or "neutral"

    # Trader analysis (for copytrade) - reserved for future use

    # Resolution timing
    window_close_time: int | None = None  # When the 5-min window closes (unix)
    resolution_time: int | None = None  # When market actually resolved (unix)
    resolution_delay_seconds: float | None = None  # Seconds between close and resolution

    # Outcome analysis
    price_at_close: float | None = None  # Our direction's price when window closed
    final_price: float | None = None  # Our direction's price after resolution (0 or 1)

    # On-chain data (from Polygonscan)
    block_number: int | None = None
    gas_used: int | None = None
    tx_fee_matic: float | None = None
    on_chain_timestamp: int | None = None

    # Delay model breakdown (for analysis)
    delay_model_breakdown: dict | None = None

    # Settlement status tracking
    settlement_status: str = "pending"  # "pending", "settled", or "force_exit"
    force_exit_reason: str | None = None  # "insufficient_bankroll" or "shutdown" (only when force_exit)

    # === LIMIT ORDER FIELDS ===
    order_type: str = "market"  # "market" | "limit"
    limit_price: float | None = None  # Requested limit price (GTC orders)
    limit_order_id: str | None = None  # Live CLOB order ID for cancellation
    cancelled_at: int | None = None  # ms timestamp when limit was cancelled
    missed: bool = False  # True if limit order expired unfilled

    # === GATE STATE ===
    gate_name: str = ""  # "none", "trend_filter", "vol_accel", "trend_vol_accel"
    gate_boosted: bool = False  # True if VolAccelGate increased size above base
    gate_skipped: bool = False  # True if gate was active but Binance fetch failed

    # Fields that are only valid during pending state and should not be persisted to JSON
    TRANSIENT_FIELDS = {"current_price", "unrealized_pnl", "implied_outcome"}

    def to_json_dict(self) -> dict:
        """Convert to dict for JSON, using nested structure."""
        return self.to_nested_json()

    def to_nested_json(self) -> dict:
        """Convert trade to nested JSON structure for clean, organized storage."""
        # Generate unique trade ID
        trade_id = f"{self.timestamp}_{self.executed_at}_{self.direction}"

        # === MARKET ===
        market = {
            "timestamp": self.timestamp,
            "slug": self.market_slug,
            "window_close": self.window_close_time or (self.timestamp + 300),
            "volume": self.market_volume,
        }

        # === POSITION ===
        position = {
            "direction": self.direction,
            "amount": self.amount,
            "requested_amount": self.requested_amount or self.amount,
            "shares": self.shares_bought,
        }

        # === EXECUTION ===
        execution = {
            "timestamp": self.executed_at,
            "entry_price": self.entry_price,
            "fill_price": self.execution_price if self.execution_price > 0 else self.entry_price,
            "spread": self.spread,
            "slippage_pct": self.slippage_pct,
            "fill_pct": self.fill_pct,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "price_movement_pct": self.price_movement_pct,
        }

        # === FEES ===
        fees = {
            "rate_bps": self.fee_rate_bps,
            "pct": self.fee_pct,
            "amount": self.fee_amount,
        }

        # === COPYTRADE (only if this is a copytrade) ===
        copytrade = None
        if self.strategy == "copytrade" and self.copied_from:
            copytrade = {
                "wallet": self.copied_from,
                "name": self.trader_name,
                "direction": self.trader_direction,
                "amount": self.trader_amount,
                "price": self.trader_price,
                "timestamp": self.trader_timestamp,
                "delay_ms": self.copy_delay_ms,
                "delay_impact_pct": self.delay_impact_pct,
                "delay_breakdown": self.delay_model_breakdown,
            }

        # === SETTLEMENT ===
        settlement = {
            "status": self.settlement_status,
            "outcome": self.outcome,
            "won": self.won,
            "timestamp": self.settled_at,
            "resolution_delay_sec": self.resolution_delay_seconds,
            "price_at_close": self.price_at_close,
            "gross_payout": self.gross_payout,
            "gross_profit": self.gross_profit,
            "fee_amount": self.fee_amount,
            "net_profit": self.net_profit,
        }
        # Add force_exit_reason if applicable
        if self.settlement_status == "force_exit":
            settlement["force_exit_reason"] = self.force_exit_reason

        # === CONTEXT ===
        mode = "paper" if self.paper else "live"
        context = {
            "strategy": self.strategy,
            "mode": mode,
            "market_bias": self.market_bias or "neutral",
        }

        # === SESSION ===
        session = {
            "trade_number": self.session_trade_number or 1,
            "wins_before": self.session_wins_before or 0,
            "losses_before": self.session_losses_before or 0,
            "pnl_before": self.session_pnl_before or 0.0,
            "bankroll_before": self.bankroll_before or 0.0,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
        }

        # === TIMING ===
        timing = {
            "hour_utc": self.hour_utc if self.hour_utc is not None else 0,
            "minute": self.minute_of_hour if self.minute_of_hour is not None else 0,
            "day_of_week": self.day_of_week if self.day_of_week is not None else 0,
            "seconds_into_window": self.seconds_into_window if self.seconds_into_window is not None else 0,
        }

        # === ON-CHAIN (reserved for future live trading) ===
        on_chain = {
            "block_number": self.block_number,
            "gas_used": self.gas_used,
            "tx_fee_matic": self.tx_fee_matic,
            "timestamp": self.on_chain_timestamp,
        }

        # Build final structure
        result = {
            "id": trade_id,
            "market": market,
            "position": position,
            "execution": execution,
            "fees": fees,
        }

        # Only include copytrade if present
        if copytrade:
            result["copytrade"] = copytrade

        result["settlement"] = settlement
        result["context"] = context
        result["session"] = session
        result["timing"] = timing
        result["on_chain"] = on_chain

        # === GATE (only include when a gate was active) ===
        if self.gate_name:
            result["gate"] = {
                "name": self.gate_name,
                "boosted": self.gate_boosted,
                "skipped": self.gate_skipped,
            }

        return result

    @classmethod
    def from_nested_json(cls, data: dict) -> "Trade":
        """Create a Trade from nested JSON structure."""
        market = data.get("market", {})
        position = data.get("position", {})
        execution = data.get("execution", {})
        fees = data.get("fees", {})
        copytrade = data.get("copytrade", {})
        settlement = data.get("settlement", {})
        context = data.get("context", {})
        session = data.get("session", {})
        timing = data.get("timing", {})
        on_chain = data.get("on_chain", {})
        gate = data.get("gate", {})

        return cls(
            # Core fields from market/position
            timestamp=market.get("timestamp", 0),
            market_slug=market.get("slug", ""),
            direction=position.get("direction", ""),
            amount=position.get("amount", 0.0),
            entry_price=execution.get("entry_price", 0.5),
            streak_length=0,  # Not stored in nested format
            confidence=0.6,  # Not stored in nested format
            paper=context.get("mode", "paper") == "paper",
            # Resolution fields from settlement
            outcome=settlement.get("outcome"),
            pnl=settlement.get("net_profit", 0.0),
            order_id=None,  # Not stored in nested format for paper
            settled_at=settlement.get("timestamp"),
            won=settlement.get("won"),
            # Settlement breakdown
            shares_bought=position.get("shares", 0.0),
            gross_payout=settlement.get("gross_payout", 0.0),
            gross_profit=settlement.get("gross_profit", 0.0),
            fee_amount=settlement.get("fee_amount", fees.get("amount", 0.0)),
            net_profit=settlement.get("net_profit", 0.0),
            # Copytrade fields
            copied_from=copytrade.get("wallet") if copytrade else None,
            trader_name=copytrade.get("name") if copytrade else None,
            trader_direction=copytrade.get("direction") if copytrade else None,
            trader_amount=copytrade.get("amount") if copytrade else None,
            trader_price=copytrade.get("price") if copytrade else None,
            trader_timestamp=copytrade.get("timestamp") if copytrade else None,
            executed_at=execution.get("timestamp"),
            copy_delay_ms=copytrade.get("delay_ms") if copytrade else None,
            market_price_at_copy=execution.get("entry_price"),
            # Strategy
            strategy=context.get("strategy", "streak"),
            # Simulation fields
            fee_rate_bps=fees.get("rate_bps", 0),
            fee_pct=fees.get("pct", 0.0),
            spread=execution.get("spread", 0.0),
            slippage_pct=execution.get("slippage_pct", 0.0),
            execution_price=execution.get("fill_price", 0.0),
            fill_pct=execution.get("fill_pct", 100.0),
            delay_impact_pct=copytrade.get("delay_impact_pct", 0.0) if copytrade else 0.0,
            requested_amount=position.get("requested_amount", position.get("amount", 0.0)),
            # Price movement
            price_at_signal=execution.get("entry_price", 0.0),
            price_at_execution=execution.get("fill_price", 0.0),
            price_movement_pct=execution.get("price_movement_pct", 0.0),
            # Market context
            market_volume=market.get("volume", 0.0),
            best_bid=execution.get("best_bid", 0.0),
            best_ask=execution.get("best_ask", 0.0),
            # Order status -> settlement_status
            order_status="pending",
            # Pattern analysis fields
            hour_utc=timing.get("hour_utc", 0),
            minute_of_hour=timing.get("minute", 0),
            day_of_week=timing.get("day_of_week", 0),
            seconds_into_window=timing.get("seconds_into_window", 0),
            # Session tracking
            session_trade_number=session.get("trade_number", 1),
            session_wins_before=session.get("wins_before", 0),
            session_losses_before=session.get("losses_before", 0),
            session_pnl_before=session.get("pnl_before", 0.0),
            bankroll_before=session.get("bankroll_before", 0.0),
            consecutive_wins=session.get("consecutive_wins", 0),
            consecutive_losses=session.get("consecutive_losses", 0),
            # Market context
            opposite_price=None,
            price_ratio=None,
            market_bias=context.get("market_bias", "neutral"),
            # Resolution timing
            window_close_time=market.get("window_close"),
            resolution_time=settlement.get("timestamp"),
            resolution_delay_seconds=settlement.get("resolution_delay_sec"),
            price_at_close=settlement.get("price_at_close"),
            final_price=1.0 if settlement.get("won") else 0.0 if settlement.get("won") is False else None,
            # On-chain data
            block_number=on_chain.get("block_number"),
            gas_used=on_chain.get("gas_used"),
            tx_fee_matic=on_chain.get("tx_fee_matic"),
            on_chain_timestamp=on_chain.get("timestamp"),
            # Delay model breakdown
            delay_model_breakdown=copytrade.get("delay_breakdown") if copytrade else None,
            # Settlement status
            settlement_status=settlement.get("status", "pending"),
            force_exit_reason=settlement.get("force_exit_reason"),
            # Gate state
            gate_name=gate.get("name", ""),
            gate_boosted=gate.get("boosted", False),
            gate_skipped=gate.get("skipped", False),
        )

    def to_history_dict(self) -> dict:
        """Convert trade to a detailed history dictionary."""
        exec_time = (
            datetime.fromtimestamp(self.executed_at / 1000, tz=LOCAL_TZ).strftime(f"%Y-%m-%d %H:%M:%S {TIMEZONE_NAME}")
            if self.executed_at
            else "N/A"
        )

        settle_time = (
            datetime.fromtimestamp(self.settled_at / 1000, tz=LOCAL_TZ).strftime(f"%Y-%m-%d %H:%M:%S {TIMEZONE_NAME}")
            if self.settled_at
            else "Pending"
        )

        return {
            # Identification
            "market": self.market_slug,
            "strategy": self.strategy,
            "mode": "PAPER" if self.paper else "LIVE",
            # Timing
            "executed_at": exec_time,
            "settled_at": settle_time,
            "copy_delay_ms": self.copy_delay_ms,
            # Position
            "direction": self.direction.upper(),
            "requested_amount": round(self.requested_amount, 2),
            "filled_amount": round(self.amount, 2),
            "fill_pct": round(self.fill_pct, 1),
            # Prices
            "price_at_signal": round(self.price_at_signal, 4),
            "entry_price": round(self.entry_price, 4),
            "execution_price": round(self.execution_price, 4),
            "price_movement_pct": round(self.price_movement_pct, 2),
            # Costs
            "spread_cents": round(self.spread * 100, 1),
            "slippage_pct": round(self.slippage_pct, 2),
            "delay_impact_pct": round(self.delay_impact_pct, 2),
            "fee_pct": round(self.fee_pct * 100, 2),
            # Shares
            "shares_bought": round(self.shares_bought, 2),
            # Result
            "outcome": self.outcome.upper() if self.outcome else "PENDING",
            "won": self.won,
            "gross_profit": round(self.gross_profit, 2),
            "fee_amount": round(self.fee_amount, 2),
            "net_pnl": round(self.pnl, 2),
            # Copytrade specific
            "copied_from": self.trader_name if self.strategy == "copytrade" else None,
            "trader_price": round(self.trader_price, 4) if self.trader_price else None,
            "trader_amount": round(self.trader_amount, 2) if self.trader_amount else None,
        }

    def summary(self) -> str:
        """Return a one-line summary of the trade."""
        status = "✓ WON" if self.won else "✗ LOST" if self.won is False else "⏳ PENDING"
        return (
            f"{self.direction.upper()} ${self.amount:.2f} @ {self.execution_price:.3f} "
            f"| {status} | PnL: ${self.pnl:+.2f}"
        )


@dataclass
class TradingState:
    """Persistent state across bot restarts."""

    trades: list[Trade] = field(default_factory=list)
    daily_bets: int = 0
    daily_pnl: float = 0.0
    last_reset_date: str = ""
    bankroll: float = field(default_factory=lambda: Config.INITIAL_BANKROLL)

    # Track which trades have been saved to full history
    _saved_trade_ids: set = field(default_factory=set)
    _last_saved_trade_id: str = ""

    def reset_daily_if_needed(self):
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self.last_reset_date != today:
            self.daily_bets = 0
            self.daily_pnl = 0.0
            self.last_reset_date = today

    def can_trade(self, bet_size: float = 0) -> tuple[bool, str]:
        self.reset_daily_if_needed()
        if self.daily_bets >= Config.MAX_DAILY_BETS:
            return False, f"Max daily bets reached ({Config.MAX_DAILY_BETS})"
        if self.daily_pnl <= -Config.MAX_DAILY_LOSS:
            return False, f"Max daily loss reached (${Config.MAX_DAILY_LOSS})"
        if self.bankroll < Config.MIN_BET:
            return False, f"Bankroll too low (${self.bankroll:.2f} < ${Config.MIN_BET:.2f})"
        if bet_size > 0 and self.bankroll < bet_size:
            return False, f"Insufficient bankroll: ${self.bankroll:.2f} < ${bet_size:.2f} bet"
        return True, "OK"

    def record_trade(self, trade: Trade):
        self.trades.append(trade)
        self.daily_bets += 1

    def settle_trade(self, trade: Trade, outcome: str, market: "Market | None" = None):
        """Settle a trade and calculate all P&L details.

        Args:
            trade: The trade to settle
            outcome: The market outcome ("up" or "down")
            market: Optional market object for resolution timing data
        """
        trade.outcome = outcome
        trade.won = trade.direction == outcome
        trade.settled_at = int(time.time() * 1000)
        trade.settlement_status = "settled"

        # Resolution timing
        resolution_time = int(time.time())
        trade.resolution_time = resolution_time
        if trade.window_close_time:
            trade.resolution_delay_seconds = resolution_time - trade.window_close_time

        # Final price is always deterministic: 1.0 if won, 0.0 if lost
        trade.final_price = 1.0 if trade.won else 0.0

        # Price at close from market data if available
        if market:
            if trade.direction == "up":
                trade.price_at_close = market.up_price
            else:
                trade.price_at_close = market.down_price

        # Use execution price (includes slippage) if available, else entry_price
        exec_price = trade.execution_price if trade.execution_price > 0 else trade.entry_price

        # Calculate shares bought
        trade.shares_bought = trade.amount / exec_price if exec_price > 0 else 0

        if trade.won:
            # Win: receive $1 per share
            trade.gross_payout = trade.shares_bought  # $1 per share on win
            trade.gross_profit = trade.gross_payout - trade.amount

            # Apply fee to the profit (fee is on proceeds, not principal)
            fee_pct = trade.fee_pct if trade.fee_pct > 0 else 0.0
            trade.fee_amount = trade.gross_profit * fee_pct if trade.gross_profit > 0 else 0.0

            trade.net_profit = trade.gross_profit - trade.fee_amount
            trade.pnl = trade.net_profit
        else:
            # Loss: lose the entire amount
            trade.gross_payout = 0.0
            trade.gross_profit = -trade.amount
            trade.fee_amount = 0.0  # No fee on losses
            trade.net_profit = -trade.amount
            trade.pnl = -trade.amount

        self.daily_pnl += trade.pnl
        self.bankroll += trade.pnl

    def mark_pending_as_force_exit(self, reason: str):
        """Mark all pending trades as force_exit before shutdown.

        Args:
            reason: "insufficient_bankroll" or "shutdown"
        """
        for trade in self.trades:
            if trade.settlement_status == "pending" and trade.outcome is None:
                trade.settlement_status = "force_exit"
                trade.force_exit_reason = reason

    def save(self):
        """Save current state and append new trades to full history."""
        # Save working state (recent trades for fast loading) using nested format
        data = {
            "trades": [t.to_nested_json() for t in self.trades[-100:]],  # keep last 100 for working state
            "daily_bets": self.daily_bets,
            "daily_pnl": self.daily_pnl,
            "last_reset_date": self.last_reset_date,
            "bankroll": self.bankroll,
            "last_trade_id": self._last_saved_trade_id,
        }
        with open(Config.TRADES_FILE, "w") as f:
            json.dump(data, f, indent=2)

        # Append new trades to full history file (never truncated)
        self._append_to_full_history()

        # Update any settled trades in full history
        self._update_settled_trades_in_history()

    def _append_to_full_history(self):
        """Append only new trades to the full history file."""
        history_file = Config.HISTORY_FILE

        # Find trades that haven't been saved yet
        new_trades = []
        for t in self.trades:
            trade_id = f"{t.timestamp}_{t.executed_at}_{t.direction}"
            if trade_id not in self._saved_trade_ids:
                new_trades.append(t)
                self._saved_trade_ids.add(trade_id)
                self._last_saved_trade_id = trade_id

        if not new_trades:
            return

        # Load existing history or create new
        existing = []
        if os.path.exists(history_file):
            try:
                with open(history_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, Exception):
                existing = []

        # Append new trades (excluding transient fields)
        for t in new_trades:
            existing.append(t.to_json_dict())

        # Save full history
        with open(history_file, "w") as f:
            json.dump(existing, f, indent=2)

        if new_trades:
            print(f"[history] Appended {len(new_trades)} trade(s) to {history_file} (total: {len(existing)})")

    def _update_settled_trades_in_history(self):
        """Update settled trades in the full history file (nested format)."""
        history_file = Config.HISTORY_FILE

        if not os.path.exists(history_file):
            return

        # Find settled or force_exit trades that need updating
        settled_trades = {
            f"{t.timestamp}_{t.executed_at}_{t.direction}": t
            for t in self.trades
            if t.settlement_status in ("settled", "force_exit")
        }

        if not settled_trades:
            return

        # Load existing history
        try:
            with open(history_file) as f:
                history = json.load(f)
        except (json.JSONDecodeError, Exception):
            return

        # Update settled trades in history (nested format)
        updated_count = 0
        for i, entry in enumerate(history):
            # Get trade ID from nested format (or reconstruct from old format for migration)
            trade_id = entry.get("id")
            if not trade_id:
                # Fallback for old flat format during migration
                market = entry.get("market", {})
                position = entry.get("position", {})
                execution = entry.get("execution", {})
                ts = market.get("timestamp") or entry.get("timestamp")
                exec_at = execution.get("timestamp") or entry.get("executed_at")
                direction = position.get("direction") or entry.get("direction")
                if ts and exec_at and direction:
                    trade_id = f"{ts}_{exec_at}_{direction}"

            if not trade_id:
                continue

            # Check if this trade has been settled/force_exit but history entry is not
            settlement = entry.get("settlement", {})
            current_status = settlement.get("status", "pending")

            if trade_id in settled_trades and current_status == "pending":
                settled_trade = settled_trades[trade_id]

                # Update settlement object in nested structure
                history[i]["settlement"] = {
                    "status": settled_trade.settlement_status,
                    "outcome": settled_trade.outcome,
                    "won": settled_trade.won,
                    "timestamp": settled_trade.settled_at,
                    "resolution_delay_sec": settled_trade.resolution_delay_seconds,
                    "price_at_close": settled_trade.price_at_close,
                    "gross_payout": settled_trade.gross_payout,
                    "gross_profit": settled_trade.gross_profit,
                    "fee_amount": settled_trade.fee_amount,
                    "net_profit": settled_trade.net_profit,
                }

                # Add force_exit_reason if applicable
                if settled_trade.settlement_status == "force_exit":
                    history[i]["settlement"]["force_exit_reason"] = settled_trade.force_exit_reason

                # Update position.shares if it was calculated during settlement
                if settled_trade.shares_bought > 0:
                    if "position" in history[i]:
                        history[i]["position"]["shares"] = settled_trade.shares_bought

                updated_count += 1

        # Save if any updates were made
        if updated_count > 0:
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)
            print(f"[history] Updated {updated_count} settled trade(s) in {history_file}")

    def export_history_json(self, filepath: str = "trade_history.json"):
        """Export full trade history to JSON file."""
        history = [t.to_history_dict() for t in self.trades]
        with open(filepath, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Exported {len(history)} trades to {filepath}")

    def export_history_csv(self, filepath: str = "trade_history.csv"):
        """Export trade history to CSV file."""
        import csv

        if not self.trades:
            print("No trades to export")
            return

        history = [t.to_history_dict() for t in self.trades]
        fieldnames = history[0].keys()

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"Exported {len(history)} trades to {filepath}")

    def print_history(self, limit: int = 20, update_unrealized: bool = True):
        """Print recent trade history to console."""
        trades = self.trades[-limit:]
        if not trades:
            print("No trade history")
            return

        # Update unrealized PnL for pending trades
        if update_unrealized:
            self.update_unrealized_pnl()

        print(f"\n{'=' * 80}")
        print(f"TRADE HISTORY (last {len(trades)} trades) - {TIMEZONE_NAME}")
        print(f"{'=' * 80}")

        for i, t in enumerate(trades, 1):
            exec_time = (
                datetime.fromtimestamp(t.executed_at / 1000, tz=LOCAL_TZ).strftime("%m/%d %H:%M")
                if t.executed_at
                else "N/A"
            )

            status = "✓" if t.won else "✗" if t.won is False else "⏳"
            strategy_icon = "📋" if t.strategy == "copytrade" else "📈"

            print(f"\n{i}. {strategy_icon} {exec_time} | {t.market_slug}")
            print(f"   Position: {t.direction.upper()} ${t.amount:.2f} @ {t.execution_price:.3f}")

            # Show costs
            costs = []
            if t.fee_pct > 0:
                costs.append(f"Fee: {t.fee_pct:.2%}")
            if t.slippage_pct > 0:
                costs.append(f"Slip: {t.slippage_pct:.2f}%")
            if t.delay_impact_pct > 0:
                costs.append(f"Delay: +{t.delay_impact_pct:.2f}%")
            if costs:
                print(f"   Costs: {' | '.join(costs)}")

            # Show result
            if t.outcome:
                print(
                    f"   Result: {status} {t.outcome.upper()} | "
                    f"Gross: ${t.gross_profit:+.2f} | Fee: ${t.fee_amount:.2f} | "
                    f"Net: ${t.pnl:+.2f}"
                )
            else:
                # Show unrealized PnL for pending trades
                if t.unrealized_pnl is not None and t.current_price is not None:
                    # Show if we're likely winning or losing
                    likely = "LIKELY WIN" if t.direction == t.implied_outcome else "LIKELY LOSS"
                    print(
                        f"   Result: {status} PENDING | "
                        f"Price: {t.current_price:.2f} ({likely}) | "
                        f"Est. PnL: ${t.unrealized_pnl:+.2f}"
                    )
                else:
                    print(f"   Result: {status} PENDING")

            # Copytrade details
            if t.strategy == "copytrade" and t.trader_name:
                print(
                    f"   Copied: {t.trader_name} (${t.trader_amount:.2f} @ {t.trader_price:.3f}) | "
                    f"Delay: {t.copy_delay_ms}ms"
                )

        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        realized_pnl = sum(t.pnl for t in trades if t.outcome)
        unrealized_pnl = sum(t.unrealized_pnl for t in trades if t.outcome is None and t.unrealized_pnl is not None)
        total_pnl = realized_pnl + unrealized_pnl
        wins = sum(1 for t in trades if t.won is True)
        losses = sum(1 for t in trades if t.won is False)
        pending_trades = [t for t in trades if t.outcome is None]
        pending_count = len(pending_trades)
        total_fees = sum(t.fee_amount for t in trades if t.outcome)

        win_rate_str = f"Win Rate: {wins / (wins + losses) * 100:.1f}%" if wins + losses > 0 else "N/A"
        print(f"Trades: {wins}W / {losses}L / {pending_count}P | {win_rate_str}")
        print(f"Realized P&L: ${realized_pnl:+.2f} | Fees Paid: ${total_fees:.2f}")
        if pending_count > 0 and unrealized_pnl != 0:
            print(f"Unrealized P&L: ${unrealized_pnl:+.2f} (from {pending_count} pending)")
            print(f"Total P&L (est): ${total_pnl:+.2f}")
        print(f"Current Bankroll: ${self.bankroll:.2f}")
        print(f"{'=' * 80}\n")

    def update_unrealized_pnl(self):
        """Update unrealized PnL for all pending trades based on current market prices."""
        pending = [t for t in self.trades if t.outcome is None]
        if not pending:
            return

        client = PolymarketClient()

        for trade in pending:
            try:
                market = client.get_market(trade.timestamp)
                if not market:
                    continue

                # Get current price for our direction
                if trade.direction == "up":
                    current_price = market.up_price
                else:
                    current_price = market.down_price

                trade.current_price = current_price

                # Implied outcome based on which side has higher probability
                if market.up_price > market.down_price:
                    trade.implied_outcome = "up"
                elif market.down_price > market.up_price:
                    trade.implied_outcome = "down"
                else:
                    trade.implied_outcome = None

                # Calculate unrealized PnL
                # If we win: receive $1 per share, minus fees
                # If we lose: lose entire amount
                exec_price = trade.execution_price if trade.execution_price > 0 else trade.entry_price
                shares = trade.amount / exec_price if exec_price > 0 else 0

                # Expected value = (prob of win * win payout) + (prob of lose * lose payout)
                # Win payout = shares - amount - fees
                # Lose payout = -amount
                win_prob = current_price
                lose_prob = 1 - current_price

                gross_win = shares - trade.amount
                fee_on_win = gross_win * trade.fee_pct if gross_win > 0 else 0
                net_win = gross_win - fee_on_win

                # Unrealized PnL = expected value
                trade.unrealized_pnl = (win_prob * net_win) + (lose_prob * (-trade.amount))

            except Exception as e:
                print(f"[unrealized] Error updating {trade.market_slug}: {e}")

    def get_statistics(self, update_unrealized: bool = True) -> dict:
        """Get comprehensive trading statistics."""
        # Update unrealized PnL for pending trades
        if update_unrealized:
            self.update_unrealized_pnl()

        settled = [t for t in self.trades if t.outcome]
        pending = [t for t in self.trades if t.outcome is None]
        wins = [t for t in settled if t.won]
        losses = [t for t in settled if not t.won]

        realized_pnl = sum(t.pnl for t in settled)
        unrealized_pnl = sum(t.unrealized_pnl for t in pending if t.unrealized_pnl is not None)

        return {
            "total_trades": len(self.trades),
            "settled_trades": len(settled),
            "pending_trades": len(pending),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(settled) * 100 if settled else 0,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
            "total_fees_paid": sum(t.fee_amount for t in settled),
            "total_gross_profit": sum(t.gross_profit for t in settled),
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "largest_win": max((t.pnl for t in wins), default=0),
            "largest_loss": min((t.pnl for t in losses), default=0),
            "avg_slippage_pct": sum(t.slippage_pct for t in settled) / len(settled) if settled else 0,
            "avg_fee_pct": sum(t.fee_pct for t in settled) / len(settled) * 100 if settled else 0,
            "avg_delay_impact_pct": sum(t.delay_impact_pct for t in settled) / len(settled) if settled else 0,
            "bankroll": self.bankroll,
        }

    @classmethod
    def load(cls) -> "TradingState":
        state = cls()

        # Load working state
        if os.path.exists(Config.TRADES_FILE):
            try:
                with open(Config.TRADES_FILE) as f:
                    data = json.load(f)

                # Load trades - detect nested vs flat format
                trades_data = data.get("trades", [])
                loaded_trades = []
                for t in trades_data:
                    # Nested format has "id" field, flat format has "timestamp" at root
                    if "id" in t or "market" in t:
                        loaded_trades.append(Trade.from_nested_json(t))
                    else:
                        # Legacy flat format - skip (starting fresh per plan)
                        print("[trader] Skipping old format trade, starting fresh")
                        continue
                state.trades = loaded_trades

                state.daily_bets = data.get("daily_bets", 0)
                state.daily_pnl = data.get("daily_pnl", 0.0)
                state.last_reset_date = data.get("last_reset_date", "")
                state.bankroll = data.get("bankroll", Config.INITIAL_BANKROLL)
                state._last_saved_trade_id = data.get("last_trade_id", "")
            except Exception as e:
                print(f"[trader] Error loading state: {e}")

        # Load saved trade IDs from full history to avoid duplicates
        history_file = Config.HISTORY_FILE
        if os.path.exists(history_file):
            try:
                with open(history_file) as f:
                    history = json.load(f)
                for t in history:
                    # Handle both nested and flat format for trade ID extraction
                    if "id" in t:
                        trade_id = t["id"]
                    else:
                        # Legacy format
                        trade_id = f"{t.get('timestamp')}_{t.get('executed_at')}_{t.get('direction')}"
                    state._saved_trade_ids.add(trade_id)
                print(f"[history] Loaded {len(state._saved_trade_ids)} trades from history")
            except Exception as e:
                print(f"[history] Error loading history: {e}")

        return state

    @classmethod
    def backfill_settlements(cls) -> tuple[int, int]:
        """Backfill settlement data for unsettled trades by querying markets.

        Works with nested JSON format. Returns tuple of (updated_count, remaining_count).
        """
        history_file = Config.HISTORY_FILE
        if not os.path.exists(history_file):
            print("[backfill] No history file found")
            return 0, 0

        # Load history
        try:
            with open(history_file) as f:
                history = json.load(f)
        except Exception as e:
            print(f"[backfill] Error loading history: {e}")
            return 0, 0

        # Find unsettled trades (nested format)
        unsettled = []
        for i, t in enumerate(history):
            settlement = t.get("settlement", {})
            if settlement.get("status", "pending") == "pending":
                unsettled.append((i, t))

        if not unsettled:
            print("[backfill] No unsettled trades found")
            return 0, 0

        print(f"[backfill] Found {len(unsettled)} unsettled trades, querying markets...")

        client = PolymarketClient()
        updated_count = 0
        still_pending = 0

        for idx, entry in unsettled:
            # Get market timestamp from nested format
            market_obj = entry.get("market", {})
            market_ts = market_obj.get("timestamp")
            if not market_ts:
                continue

            market = client.get_market(market_ts)
            if not market:
                print(f"[backfill] Market not found for ts={market_ts}")
                still_pending += 1
                continue

            if not market.closed or not market.outcome:
                print(f"[backfill] Market {market.slug} not yet settled")
                still_pending += 1
                continue

            # Calculate settlement from nested format
            position = entry.get("position", {})
            execution = entry.get("execution", {})
            fees = entry.get("fees", {})

            direction = position.get("direction")
            outcome = market.outcome
            won = direction == outcome
            amount = position.get("amount", 0)
            exec_price = execution.get("fill_price") or execution.get("entry_price", 0.5)
            fee_pct = fees.get("pct", 0)

            shares_bought = amount / exec_price if exec_price > 0 else 0

            if won:
                gross_payout = shares_bought  # $1 per share
                gross_profit = gross_payout - amount
                fee_amount = gross_profit * fee_pct if gross_profit > 0 else 0
                net_profit = gross_profit - fee_amount
            else:
                gross_payout = 0.0
                gross_profit = -amount
                fee_amount = 0.0
                net_profit = -amount

            # Update settlement in nested structure
            history[idx]["settlement"] = {
                "status": "settled",
                "outcome": outcome,
                "won": won,
                "timestamp": int(time.time() * 1000),
                "resolution_delay_sec": None,
                "price_at_close": market.up_price if direction == "up" else market.down_price,
                "gross_payout": gross_payout,
                "gross_profit": gross_profit,
                "fee_amount": fee_amount,
                "net_profit": net_profit,
            }

            # Update position shares
            if "position" in history[idx]:
                history[idx]["position"]["shares"] = shares_bought

            emoji = "✓" if won else "✗"
            print(
                f"[backfill] {emoji} {market.slug}: {direction.upper()} -> {outcome.upper()} | PnL: ${net_profit:+.2f}"
            )
            updated_count += 1

        # Save updated history
        if updated_count > 0:
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)
            print(f"[backfill] Updated {updated_count} trades in {history_file}")

        return updated_count, still_pending

    @classmethod
    def load_full_history(cls) -> "TradingState":
        """Load complete trade history from the full history file."""
        state = cls()
        history_file = Config.HISTORY_FILE

        if os.path.exists(history_file):
            try:
                with open(history_file) as f:
                    history = json.load(f)
                loaded_trades = []
                for t in history:
                    # Nested format has "id" field
                    if "id" in t or "market" in t:
                        loaded_trades.append(Trade.from_nested_json(t))
                    else:
                        # Legacy flat format - skip
                        continue
                state.trades = loaded_trades
                print(f"[history] Loaded {len(state.trades)} trades from full history")
            except Exception as e:
                print(f"[history] Error loading full history: {e}")

        # Also load current bankroll from working state
        if os.path.exists(Config.TRADES_FILE):
            try:
                with open(Config.TRADES_FILE) as f:
                    data = json.load(f)
                state.bankroll = data.get("bankroll", Config.INITIAL_BANKROLL)
            except Exception:
                pass

        return state


class PaperTrader:
    """Paper trading — logs trades without executing, with realistic simulation."""

    def __init__(self, market_cache=None):
        """Initialize paper trader.

        Args:
            market_cache: Optional MarketDataCache for faster orderbook lookups
        """
        self._client = PolymarketClient(timeout=Config.REST_TIMEOUT)
        self._market_cache = market_cache

    def place_bet(
        self,
        market: Market,
        direction: str,
        amount: float,
        confidence: float,
        streak_length: int,
        **kwargs,  # copytrade fields
    ) -> Trade | None:
        """Place a simulated bet with realistic fees, slippage, and fill simulation.

        Returns None if order is rejected (e.g., below minimum size).
        """
        # Validate minimum order size
        if amount < Config.MIN_BET:
            print(f"[PAPER] ❌ Order rejected: ${amount:.2f} below minimum ${Config.MIN_BET:.2f}")
            return None

        entry_price = market.up_price if direction == "up" else market.down_price
        executed_at = int(time.time() * 1000)  # milliseconds

        # Get token ID for the direction we're betting on
        token_id = market.up_token_id if direction == "up" else market.down_token_id

        # Default simulation values
        fee_rate_bps = 0
        fee_pct = 0.0
        spread = 0.0
        slippage_pct = 0.0
        fill_pct = 100.0
        delay_impact_pct = 0.0
        delay_breakdown = None
        execution_price = entry_price if entry_price > 0 else 0.5
        best_bid = 0.0
        best_ask = 0.0
        market_volume = market.volume if hasattr(market, "volume") else 0.0

        # Price at signal (before any processing)
        price_at_signal = entry_price

        # Get copy delay if this is a copytrade
        copy_delay_ms = kwargs.get("copy_delay_ms", 0)
        precomputed_execution = kwargs.pop("precomputed_execution", None)

        # Use fee rate from market data (already fetched from Gamma API)
        fee_rate_bps = market.taker_fee_bps if hasattr(market, "taker_fee_bps") else 200
        fee_pct = self._client.calculate_fee(execution_price, fee_rate_bps)

        # Query orderbook for realistic simulation (or use precomputed data)
        if precomputed_execution:
            execution_price = precomputed_execution.get("execution_price", execution_price)
            spread = precomputed_execution.get("spread", spread)
            slippage_pct = precomputed_execution.get("slippage_pct", slippage_pct)
            fill_pct = precomputed_execution.get("fill_pct", fill_pct)
            delay_impact_pct = precomputed_execution.get("delay_impact_pct", delay_impact_pct)
            delay_breakdown = precomputed_execution.get("delay_breakdown")
            best_bid = precomputed_execution.get("best_bid", best_bid)
            best_ask = precomputed_execution.get("best_ask", best_ask)
            if execution_price > 0:
                fee_pct = self._client.calculate_fee(execution_price, fee_rate_bps)
        elif token_id:
            try:
                # Use market cache if available (faster, WebSocket-backed)
                if self._market_cache:
                    book = self._market_cache.get_orderbook(token_id)
                else:
                    book = self._client.get_orderbook(token_id)

                if book:
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    if bids:
                        best_bid = max(float(b["price"]) for b in bids)
                    if asks:
                        best_ask = min(float(a["price"]) for a in asks)

                # Get execution price with slippage and copy delay impact
                delay_breakdown = None
                if self._market_cache:
                    (
                        exec_price,
                        spread,
                        slippage_pct,
                        fill_pct,
                        delay_impact_pct,
                        delay_breakdown,
                    ) = self._market_cache.get_execution_price(token_id, "BUY", amount, copy_delay_ms)
                else:
                    (
                        exec_price,
                        spread,
                        slippage_pct,
                        fill_pct,
                        delay_impact_pct,
                        delay_breakdown,
                    ) = self._client.get_execution_price(token_id, "BUY", amount, copy_delay_ms)

                if exec_price > 0:
                    execution_price = exec_price
                    # Recalculate fee at actual execution price
                    fee_pct = self._client.calculate_fee(execution_price, fee_rate_bps)
            except Exception as e:
                print(f"[PAPER] Warning: Could not fetch market data: {e}")

        # Calculate price movement from signal to execution
        price_movement_pct = 0.0
        if price_at_signal > 0:
            price_movement_pct = ((execution_price - price_at_signal) / price_at_signal) * 100

        # Adjust amount for partial fill
        filled_amount = amount * (fill_pct / 100.0)
        if fill_pct < 100.0:
            print(f"[PAPER] ⚠️  Partial fill: {fill_pct:.1f}% of ${amount:.2f} = ${filled_amount:.2f}")

        # === PATTERN ANALYSIS DATA ===
        # Time-based patterns
        exec_dt = datetime.fromtimestamp(executed_at / 1000, tz=UTC)
        hour_utc = exec_dt.hour
        minute_of_hour = exec_dt.minute
        day_of_week = exec_dt.weekday()  # 0=Monday, 6=Sunday

        # How far into the 5-min window are we entering?
        window_start = market.timestamp
        seconds_into_window = int(executed_at / 1000) - window_start
        window_close_time = window_start + 300  # 5 min window

        # Market context - opposite outcome price
        opposite_price = market.down_price if direction == "up" else market.up_price
        price_ratio = entry_price / opposite_price if opposite_price > 0 else 1.0

        # Market bias based on prices
        if market.up_price > 0.52:
            market_bias = "bullish"
        elif market.down_price > 0.52:
            market_bias = "bearish"
        else:
            market_bias = "neutral"

        trade = Trade(
            timestamp=market.timestamp,
            market_slug=market.slug,
            direction=direction,
            amount=filled_amount,  # Use filled amount, not requested amount
            entry_price=entry_price if entry_price > 0 else 0.5,
            streak_length=streak_length,
            confidence=confidence,
            paper=True,
            executed_at=executed_at,
            market_price_at_copy=entry_price,
            # Realistic simulation fields
            fee_rate_bps=fee_rate_bps,
            fee_pct=fee_pct,
            spread=spread,
            slippage_pct=slippage_pct,
            execution_price=execution_price,
            fill_pct=fill_pct,
            delay_impact_pct=delay_impact_pct,
            requested_amount=amount,  # Original requested amount
            # Price movement tracking
            price_at_signal=price_at_signal,
            price_at_execution=execution_price,
            price_movement_pct=price_movement_pct,
            # Market context
            market_volume=market_volume,
            best_bid=best_bid,
            best_ask=best_ask,
            # Pattern analysis fields
            hour_utc=hour_utc,
            minute_of_hour=minute_of_hour,
            day_of_week=day_of_week,
            seconds_into_window=seconds_into_window,
            window_close_time=window_close_time,
            opposite_price=opposite_price,
            price_ratio=price_ratio,
            market_bias=market_bias,
            # Delay model breakdown for analysis
            delay_model_breakdown=delay_breakdown,
            **kwargs,  # pass copytrade fields
        )

        # Log trade details with fee, spread, slippage
        spread_cents = spread * 100  # Convert to cents for display
        if kwargs.get("strategy") == "copytrade":
            trader = kwargs.get("trader_name", "unknown")
            trader_amt = kwargs.get("trader_amount", 0)
            delay_info = f" | Delay impact: +{delay_impact_pct:.2f}%" if delay_impact_pct > 0 else ""
            print(
                f"[PAPER] Copied {trader}: ${filled_amount:.2f} on {direction.upper()} @ {execution_price:.3f} "
                f"| Trader bet ${trader_amt:.2f} @ {kwargs.get('trader_price', 0):.2f}"
            )
            print(
                f"        Fee: {fee_pct:.2%} | Spread: {spread_cents:.0f}¢ | Slippage: {slippage_pct:.2f}%{delay_info}"
            )
        else:
            print(
                f"[PAPER] Bet ${filled_amount:.2f} on {direction.upper()} @ {execution_price:.3f} "
                f"| {market.title} | streak={streak_length} conf={confidence:.1%}"
            )
            print(f"        Fee: {fee_pct:.2%} | Spread: {spread_cents:.0f}¢ | Slippage: {slippage_pct:.2f}%")
        return trade

    def place_limit_bet(
        self,
        market: Market,
        direction: str,
        amount: float,
        limit_price: float,
        confidence: float,
        streak_length: int,
    ) -> Trade | None:
        """Create a pending GTC limit trade (paper simulation).

        Does NOT simulate fill immediately. Caller must poll check_limit_fill()
        and handle expiry via cancel_limit_bet().

        Returns None if below minimum order size.
        """
        if amount < Config.MIN_BET:
            print(f"[PAPER] Limit rejected: ${amount:.2f} below minimum ${Config.MIN_BET:.2f}")
            return None

        entry_price = market.up_price if direction == "up" else market.down_price
        if entry_price <= 0:
            entry_price = 0.5
        executed_at = int(time.time() * 1000)

        fee_rate_bps = market.taker_fee_bps if hasattr(market, "taker_fee_bps") else 200
        fee_pct = self._client.calculate_fee(limit_price, fee_rate_bps)

        exec_dt = datetime.fromtimestamp(executed_at / 1000, tz=UTC)
        window_start = market.timestamp
        seconds_into_window = int(executed_at / 1000) - window_start
        window_close_time = window_start + 300
        opposite_price = market.down_price if direction == "up" else market.up_price
        price_ratio = entry_price / opposite_price if opposite_price > 0 else 1.0
        if market.up_price > 0.52:
            market_bias = "bullish"
        elif market.down_price > 0.52:
            market_bias = "bearish"
        else:
            market_bias = "neutral"

        trade = Trade(
            timestamp=market.timestamp,
            market_slug=market.slug,
            direction=direction,
            amount=amount,
            entry_price=entry_price,
            streak_length=streak_length,
            confidence=confidence,
            paper=True,
            executed_at=executed_at,
            market_price_at_copy=entry_price,
            fee_rate_bps=fee_rate_bps,
            fee_pct=fee_pct,
            requested_amount=amount,
            price_at_signal=entry_price,
            market_volume=market.volume if hasattr(market, "volume") else 0.0,
            hour_utc=exec_dt.hour,
            minute_of_hour=exec_dt.minute,
            day_of_week=exec_dt.weekday(),
            seconds_into_window=seconds_into_window,
            window_close_time=window_close_time,
            opposite_price=opposite_price,
            price_ratio=price_ratio,
            market_bias=market_bias,
            # Limit order fields
            order_type="limit",
            limit_price=limit_price,
            order_status="pending",
            settlement_status="pending",
        )

        print(
            f"[PAPER] Limit bid ${amount:.2f} {direction.upper()} @ {limit_price:.4f} "
            f"(ask={entry_price:.4f}) | streak={streak_length}"
        )
        return trade

    # Maximum discount from ask that paper simulation will treat as fillable.
    # Live data showed limits >3% below ask almost never fill on the thin
    # Polymarket ETH CLOB within a 285s window.  Paper fills beyond this
    # threshold would be unrealistic and inflate paper performance vs live.
    PAPER_MAX_FILL_DISCOUNT = 0.03

    def check_limit_fill(self, trade: Trade) -> bool:
        """Check if a pending limit order has been filled by querying the orderbook.

        Returns True if the current best ask is at or below the limit price
        AND the limit is within PAPER_MAX_FILL_DISCOUNT of the ask at placement.
        The discount guard prevents the paper sim from filling deep limits that
        converge toward resolution (ask → 0) but would never match on a real CLOB.
        """
        token_id = None
        try:
            market = self._client.get_market(trade.timestamp)
            if not market:
                return False
            token_id = market.up_token_id if trade.direction == "up" else market.down_token_id
        except Exception:
            return False

        if not token_id:
            return False

        try:
            # Try spread endpoint first (fastest)
            spread = self._client.get_spread(token_id)
            if spread and spread[1] > 0:
                _bid, current_ask = spread
            else:
                current_ask_maybe = self._client.get_price(token_id, side="BUY")
                current_ask = current_ask_maybe if current_ask_maybe else 1.0
        except Exception:
            return False

        if trade.limit_price is None:
            return False

        # Guard: if our limit was placed more than PAPER_MAX_FILL_DISCOUNT below
        # the ask at placement, don't simulate a fill — this mirrors real CLOB
        # behaviour where deep limits don't get matched on thin markets.
        ask_at_placement = trade.entry_price or current_ask
        if (
            ask_at_placement > 0
            and (ask_at_placement - trade.limit_price) / ask_at_placement > self.PAPER_MAX_FILL_DISCOUNT
        ):
            return False

        if current_ask <= trade.limit_price:
            # Simulate fill at the limit price (we get the price we asked for)
            trade.execution_price = trade.limit_price
            trade.price_at_execution = trade.limit_price
            trade.order_status = "filled"
            fee_rate_bps = trade.fee_rate_bps or 200
            trade.fee_pct = self._client.calculate_fee(trade.limit_price, fee_rate_bps)
            return True
        return False

    def cancel_limit_bet(self, trade: Trade, missed_log_path: str = "missed_orders.json") -> bool:
        """Mark a limit trade as missed/expired and log it to missed_orders.json.

        Always returns True (paper cancels never fail).
        """
        trade.missed = True
        trade.cancelled_at = int(time.time() * 1000)
        trade.order_status = "cancelled"

        record = {
            "timestamp": trade.cancelled_at,
            "slug": trade.market_slug,
            "direction": trade.direction,
            "streak_length": trade.streak_length,
            "limit_price": trade.limit_price,
            "ask_at_placement": trade.entry_price,
            "discount_applied": round(trade.entry_price - (trade.limit_price or 0), 4),
            "fill_window_sec": Config.ALT_ENTRY_FILL_WINDOW_SEC,
            "cancel_failed": False,
            "paper": True,
        }

        existing: list = []
        if os.path.exists(missed_log_path):
            try:
                with open(missed_log_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.append(record)
        with open(missed_log_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(
            f"[PAPER] Limit expired: {trade.direction.upper()} @ {trade.limit_price:.4f} "
            f"(slug={trade.market_slug}) → logged to {missed_log_path}"
        )
        return True


class LiveTrader:
    """Live trading via Polymarket CLOB API.

    Supports:
    - EOA/MetaMask wallets (signature_type=0, default)
    - Magic/proxy wallets (signature_type=1, requires funder address)
    - FOK (Fill-Or-Kill) market orders for immediate execution
    - Order status tracking and confirmation
    """

    # Minimum order size in USD
    MIN_ORDER_SIZE = 1.0

    def __init__(self, market_cache=None):
        """Initialize live trader.

        Args:
            market_cache: Optional MarketDataCache for faster orderbook lookups
        """
        if not Config.PRIVATE_KEY:
            raise ValueError("PRIVATE_KEY not set in .env")

        # Validate proxy wallet config
        if Config.SIGNATURE_TYPE == 1 and not Config.FUNDER_ADDRESS:
            raise ValueError("FUNDER_ADDRESS required for proxy wallet (SIGNATURE_TYPE=1)")

        self._market_cache = market_cache
        self._init_client()
        self._circuit_breaker = CircuitBreaker(name="polymarket-clob")
        self._rate_limiter = RateLimiter(requests_per_minute=60)

    def _init_client(self):
        """Initialize py-clob-client with wallet credentials."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            # Build client based on wallet type
            if Config.SIGNATURE_TYPE == 1:
                print(f"[trader] Using proxy wallet with funder: {Config.FUNDER_ADDRESS[:10]}...")
                self.client = ClobClient(
                    host=Config.CLOB_API,
                    key=Config.PRIVATE_KEY,
                    chain_id=Config.CHAIN_ID,
                    signature_type=1,
                    funder=Config.FUNDER_ADDRESS,
                )
            else:
                self.client = ClobClient(
                    host=Config.CLOB_API,
                    key=Config.PRIVATE_KEY,
                    chain_id=Config.CHAIN_ID,
                )

            # Set API credentials — use stored keys if available, otherwise derive
            if Config.CLOB_API_KEY and Config.CLOB_API_SECRET and Config.CLOB_API_PASSPHRASE:
                from py_clob_client.clob_types import ApiCreds

                creds = ApiCreds(
                    api_key=Config.CLOB_API_KEY,
                    api_secret=Config.CLOB_API_SECRET,
                    api_passphrase=Config.CLOB_API_PASSPHRASE,
                )
                print("[trader] Using stored CLOB API credentials (CLOB_API_KEY)")
            else:
                creds = self.client.create_or_derive_api_creds()
                print("[trader] Derived CLOB API credentials from PRIVATE_KEY")
            self.client.set_api_creds(creds)

            # Sync CLOB's cached balance/allowance view from on-chain state
            try:
                from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

                params = BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,  # type: ignore[arg-type]
                    signature_type=Config.SIGNATURE_TYPE,
                )
                self.client.update_balance_allowance(params)
                result = self.client.get_balance_allowance(params)
                if isinstance(result, dict):
                    bal = float(result.get("balance", 0)) / 1_000_000
                    # API returns either "allowance" (singular) or "allowances" (dict keyed by spender)
                    raw_alw = result.get("allowance") or result.get("allowances") or 0
                    if isinstance(raw_alw, dict):
                        raw_alw = min(raw_alw.values()) if raw_alw else 0
                    alw = float(raw_alw) / 1_000_000
                    print(f"[trader] CLOB balance: ${bal:.2f}  allowance: ${alw:.2f}")
                    if alw < 1.0:
                        print("[trader] WARNING: CLOB allowance is very low — run scripts/approve_usdc.py")
            except Exception as e:
                print(f"[trader] Warning: could not sync CLOB balance/allowance: {e}")

            # Store order types and constants
            self.MarketOrderArgs = MarketOrderArgs
            self.OrderArgs = OrderArgs
            self.OrderType = OrderType
            self.BUY = BUY
            self.SELL = SELL

            wallet_type = "proxy" if Config.SIGNATURE_TYPE == 1 else "EOA"
            print(f"[trader] Live trading client initialized ({wallet_type} wallet)")

        except ImportError:
            raise ImportError("py-clob-client not installed. Run: pip install py-clob-client")
        except Exception as e:
            raise RuntimeError(f"Failed to init trading client: {e}")

    def _validate_order(self, market: Market, direction: str, amount: float) -> tuple[bool, str]:
        """Validate order parameters before submission.

        Returns:
            (is_valid, error_message)
        """
        # Check minimum order size
        if amount < self.MIN_ORDER_SIZE:
            return False, f"Order size ${amount:.2f} below minimum ${self.MIN_ORDER_SIZE:.2f}"

        # Check token ID exists
        token_id = market.up_token_id if direction == "up" else market.down_token_id
        if not token_id:
            return False, f"No token ID for {direction} side"

        # Check market is accepting orders
        if not market.accepting_orders:
            return False, f"Market {market.slug} not accepting orders"

        # Check market is not closed
        if market.closed:
            return False, f"Market {market.slug} is closed"

        return True, ""

    def _get_order_status(self, order_id: str, max_attempts: int = 5, poll_interval: float = 0.5) -> dict:
        """Poll for order status until filled or timeout.

        Args:
            order_id: Order ID to check
            max_attempts: Maximum polling attempts
            poll_interval: Seconds between polls

        Returns:
            Order status dict with keys: status, filled_size, avg_price, etc.
        """
        for _attempt in range(max_attempts):
            try:
                order = self.client.get_order(order_id)
                if not isinstance(order, dict):
                    continue
                status = order.get("status", "unknown")

                # FOK orders should be immediately filled or cancelled
                if status in ("FILLED", "MATCHED"):
                    return {
                        "status": "filled",
                        "filled_size": float(order.get("size_matched", order.get("size", 0))),
                        "avg_price": float(order.get("price", 0)),
                        "order": order,
                    }
                elif status in ("CANCELED", "CANCELLED", "EXPIRED"):
                    return {
                        "status": "cancelled",
                        "filled_size": 0,
                        "avg_price": 0,
                        "order": order,
                    }
                elif status == "LIVE":
                    # FOK should not rest on book, but check anyway
                    time.sleep(poll_interval)
                    continue
                else:
                    # Unknown status, keep polling
                    time.sleep(poll_interval)

            except Exception as e:
                print(f"[trader] Error polling order {order_id}: {e}")
                time.sleep(poll_interval)

        # Timeout - return unknown status
        return {
            "status": "unknown",
            "filled_size": 0,
            "avg_price": 0,
            "order": None,
        }

    def place_bet(
        self,
        market: Market,
        direction: str,
        amount: float,
        confidence: float,
        streak_length: int,
        **kwargs,  # copytrade fields
    ) -> Trade | None:
        """Place a live bet using FOK (Fill-Or-Kill) market order.

        FOK orders fill immediately at the best available price or are cancelled.
        This is ideal for copy trading where speed matters.

        Returns None if order is rejected (validation failed, market closed, etc.)
        """
        # Validate order parameters
        is_valid, error_msg = self._validate_order(market, direction, amount)
        if not is_valid:
            print(f"[LIVE] Order rejected: {error_msg}")
            return None

        # Precomputed execution data is only used by paper mode; discard if passed
        kwargs.pop("precomputed_execution", None)

        token_id = market.up_token_id if direction == "up" else market.down_token_id
        if token_id is None:
            return None
        entry_price = market.up_price if direction == "up" else market.down_price
        if entry_price <= 0:
            entry_price = 0.5

        executed_at = int(time.time() * 1000)  # milliseconds
        order_id = None
        order_status = "pending"
        execution_price = entry_price
        filled_amount = amount

        # Get fee rate from market
        fee_rate_bps = market.taker_fee_bps if hasattr(market, "taker_fee_bps") else 200
        fee_pct = PolymarketClient.calculate_fee(entry_price, fee_rate_bps)

        try:
            # Create FOK market order
            # For BUY orders, amount is in USD (how much to spend)
            market_order = self.MarketOrderArgs(
                token_id=token_id,
                amount=amount,  # USD amount to spend
                side=self.BUY,
                order_type=self.OrderType.FOK,  # type: ignore[invalid-argument-type]  # Fill-Or-Kill for immediate execution
            )

            # Sign and submit the order
            signed_order = self.client.create_market_order(market_order)
            response = self.client.post_order(signed_order, self.OrderType.FOK)  # type: ignore[invalid-argument-type]

            resp_dict: dict = response if isinstance(response, dict) else {}
            order_id = resp_dict.get("orderID", resp_dict.get("id", "unknown"))
            order_status = "submitted"

            # Log based on strategy type
            if kwargs.get("strategy") == "copytrade":
                trader = kwargs.get("trader_name", "unknown")
                print(
                    f"[LIVE] Copied {trader}: ${amount:.2f} on {direction.upper()} @ {entry_price:.2f} "
                    f"| order={order_id} (FOK)"
                )
            else:
                print(
                    f"[LIVE] Bet ${amount:.2f} on {direction.upper()} @ {entry_price:.2f} "
                    f"| {market.title} | order={order_id} (FOK)"
                )

            # Poll for order status (FOK should resolve quickly)
            if order_id and not order_id.startswith("FAILED"):
                status_result = self._get_order_status(order_id)
                order_status = status_result["status"]

                if order_status == "filled":
                    filled_amount = status_result["filled_size"] * status_result["avg_price"]
                    execution_price = status_result["avg_price"]
                    print(f"[LIVE] Order filled: {status_result['filled_size']:.2f} shares @ {execution_price:.3f}")
                elif order_status == "cancelled":
                    print("[LIVE] Order cancelled (FOK not filled)")
                    return None
                else:
                    print(f"[LIVE] Order status: {order_status}")

        except Exception as e:
            print(f"[LIVE] Order failed: {e}")
            order_id = f"FAILED:{e}"
            order_status = "failed"

            # Categorize the error
            category = categorize_error(e)
            if category == ErrorCategory.FATAL:
                print(f"[LIVE] Fatal error (not retryable): {e}")
                return None

        return Trade(
            timestamp=market.timestamp,
            market_slug=market.slug,
            direction=direction,
            amount=filled_amount,
            entry_price=entry_price,
            streak_length=streak_length,
            confidence=confidence,
            paper=False,
            order_id=order_id,
            executed_at=executed_at,
            market_price_at_copy=entry_price,
            # Realistic execution fields
            fee_rate_bps=fee_rate_bps,
            fee_pct=fee_pct,
            execution_price=execution_price,
            requested_amount=amount,
            price_at_signal=entry_price,
            price_at_execution=execution_price,
            **kwargs,  # pass copytrade fields
        )

    def place_limit_bet(
        self,
        market: Market,
        direction: str,
        amount: float,
        limit_price: float,
        confidence: float,
        streak_length: int,
    ) -> Trade | None:
        """Place a GTC limit order on the live CLOB.

        Returns a Trade with order_type="limit" and outcome=None.
        The caller must poll check_limit_fill() and cancel via cancel_limit_bet().
        """
        is_valid, error_msg = self._validate_order(market, direction, amount)
        if not is_valid:
            print(f"[LIVE] Limit rejected: {error_msg}")
            return None

        token_id = market.up_token_id if direction == "up" else market.down_token_id
        if token_id is None:
            return None

        entry_price = market.up_price if direction == "up" else market.down_price
        if entry_price <= 0:
            entry_price = 0.5
        executed_at = int(time.time() * 1000)
        fee_rate_bps = market.taker_fee_bps if hasattr(market, "taker_fee_bps") else 200
        fee_pct = PolymarketClient.calculate_fee(limit_price, fee_rate_bps)

        clob_order_id: str | None = None
        order_status = "pending"

        try:
            # shares = USD / price; enforce Polymarket CLOB minimum of 5 shares
            shares = round(amount / limit_price, 4) if limit_price > 0 else 0.0
            if 0 < shares < 5.0:
                shares = 5.0
                amount = round(5.0 * limit_price, 2)
            order_args = self.OrderArgs(
                token_id=token_id,
                price=limit_price,
                size=shares,
                side=self.BUY,
            )
            signed_order = self.client.create_order(order_args)
            response = with_retry(
                lambda: self.client.post_order(signed_order, self.OrderType.GTC),  # type: ignore[invalid-argument-type]
                circuit_breaker=self._circuit_breaker,
                rate_limiter=self._rate_limiter,
                max_retries=2,
                base_delay=1.0,
            )

            resp_dict: dict = response if isinstance(response, dict) else {}
            clob_order_id = resp_dict.get("orderID", resp_dict.get("id"))
            order_status = "submitted"
            print(
                f"[LIVE] Limit bid ${amount:.2f} {direction.upper()} @ {limit_price:.4f} "
                f"| streak={streak_length} | order={clob_order_id} (GTC)"
            )
        except Exception as e:
            print(f"[LIVE] Limit order failed: {e}")
            order_status = "failed"
            return None

        exec_dt = datetime.fromtimestamp(executed_at / 1000, tz=UTC)
        return Trade(
            timestamp=market.timestamp,
            market_slug=market.slug,
            direction=direction,
            amount=amount,
            entry_price=entry_price,
            streak_length=streak_length,
            confidence=confidence,
            paper=False,
            order_id=clob_order_id,
            executed_at=executed_at,
            market_price_at_copy=entry_price,
            fee_rate_bps=fee_rate_bps,
            fee_pct=fee_pct,
            requested_amount=amount,
            price_at_signal=entry_price,
            hour_utc=exec_dt.hour,
            minute_of_hour=exec_dt.minute,
            day_of_week=exec_dt.weekday(),
            seconds_into_window=int(executed_at / 1000) - market.timestamp,
            window_close_time=market.timestamp + 300,
            market_volume=market.volume if hasattr(market, "volume") else 0.0,
            order_status=order_status,
            # Limit order fields
            order_type="limit",
            limit_price=limit_price,
            limit_order_id=clob_order_id,
        )

    def check_limit_fill(self, trade: Trade) -> bool:
        """Check if a live GTC limit order has been filled.

        Returns True if filled; updates trade.execution_price on fill.
        """
        order_id = trade.limit_order_id or trade.order_id
        if not order_id:
            return False

        try:
            order = with_retry(
                lambda: self.client.get_order(order_id),
                circuit_breaker=self._circuit_breaker,
                rate_limiter=self._rate_limiter,
                max_retries=2,
                base_delay=1.0,
            )
            if not isinstance(order, dict):
                return False
            status = order.get("status", "").upper()

            if status in ("FILLED", "MATCHED"):
                size_matched = float(order.get("size_matched", order.get("size", 0)))
                avg_price = float(order.get("price", trade.limit_price or 0))
                if size_matched < float(order.get("size", size_matched)):
                    print(f"[LIVE] Warning: partial fill detected for {order_id} — size_matched={size_matched}")
                trade.execution_price = avg_price
                trade.price_at_execution = avg_price
                trade.shares_bought = size_matched
                trade.order_status = "filled"
                return True
        except Exception as e:
            print(f"[LIVE] Error checking limit fill for {order_id}: {e}")

        return False

    def cancel_limit_bet(self, trade: Trade, missed_log_path: str = "missed_orders.json") -> bool:
        """Cancel a live GTC limit order and log it as missed.

        Returns True if the CLOB cancel succeeded, False if the cancel API call failed
        (the order is still dropped from pending_limits regardless).
        """
        order_id = trade.limit_order_id or trade.order_id
        cancel_failed = False
        if order_id:
            try:
                with_retry(
                    lambda: self.client.cancel(order_id),
                    circuit_breaker=self._circuit_breaker,
                    rate_limiter=self._rate_limiter,
                    max_retries=2,
                    base_delay=1.0,
                )
            except Exception as e:
                print(f"[LIVE] Error cancelling order {order_id}: {e}")
                cancel_failed = True

        trade.missed = True
        trade.cancelled_at = int(time.time() * 1000)
        trade.order_status = "cancel_failed" if cancel_failed else "cancelled"

        record = {
            "timestamp": trade.cancelled_at,
            "slug": trade.market_slug,
            "direction": trade.direction,
            "streak_length": trade.streak_length,
            "limit_price": trade.limit_price,
            "ask_at_placement": trade.entry_price,
            "discount_applied": round(trade.entry_price - (trade.limit_price or 0), 4),
            "fill_window_sec": Config.ALT_ENTRY_FILL_WINDOW_SEC,
            "cancel_failed": cancel_failed,
            "paper": False,
        }

        existing: list = []
        if os.path.exists(missed_log_path):
            try:
                with open(missed_log_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.append(record)
        with open(missed_log_path, "w") as f:
            json.dump(existing, f, indent=2)

        if cancel_failed:
            print(
                f"[LIVE] Cancel FAILED for {trade.direction.upper()} @ {trade.limit_price:.4f} "
                f"(order={order_id}) — dropped from pending, logged cancel_failed=True"
            )
        else:
            print(
                f"[LIVE] Limit expired: {trade.direction.upper()} @ {trade.limit_price:.4f} "
                f"(order={order_id}) → logged to {missed_log_path}"
            )
        return not cancel_failed

    # Free public Polygon RPC endpoints, tried in order.
    # polygon-rpc.com now requires auth; these are open as of 2026.
    # Override all of them by setting POLYGON_RPC_URL in .env.
    _FALLBACK_RPCS = [
        "https://rpc.ankr.com/polygon",
        "https://polygon.llamarpc.com",
        "https://polygon-bor-rpc.publicnode.com",
    ]

    # Polymarket migrated from USDC.e to native USDC on Polygon (2024).
    # py-clob-client still returns the old USDC.e address from get_collateral_address().
    # We check both and sum them so the balance is correct regardless of which
    # token the user holds or which version of the exchange is in use.
    _NATIVE_USDC_POLYGON = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

    def get_wallet_balance(self) -> float | None:
        """Read actual USDC balance directly from the Polygon blockchain.

        Uses web3 balanceOf() on the USDC contract — always reflects the real
        wallet balance regardless of CLOB allowance state. Returns USD (float)
        or None if all RPC endpoints fail.

        Tries POLYGON_RPC_URL from .env first, then falls back through a list
        of free public endpoints. Wallet and USDC addresses come from the
        already-initialised py-clob-client (no extra config needed).
        """
        try:
            from web3 import Web3

            # Prefer WALLET_ADDRESS env override (proxy wallet); fall back to
            # the address derived from the signing key.
            wallet_addr = Config.WALLET_ADDRESS or self.client.get_address()
            collateral_addr = self.client.get_collateral_address()
            if not wallet_addr or not collateral_addr:
                print("[LIVE] get_wallet_balance: could not resolve wallet/collateral address")
                return None

            erc20_abi = [
                {
                    "inputs": [{"name": "account", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "stateMutability": "view",
                    "type": "function",
                },
                {
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "stateMutability": "view",
                    "type": "function",
                },
            ]

            # Build candidate list: env override first, then public fallbacks
            candidates = [Config.POLYGON_RPC_URL] + [r for r in self._FALLBACK_RPCS if r != Config.POLYGON_RPC_URL]

            # Check both USDC variants: py-clob-client uses USDC.e but Polymarket
            # migrated to native USDC in 2024. Sum both so the result is correct
            # regardless of which token the wallet holds.
            usdc_addrs = list({collateral_addr, self._NATIVE_USDC_POLYGON})

            for rpc_url in candidates:
                try:
                    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 5}))
                    wallet_cs = Web3.to_checksum_address(wallet_addr)
                    total = 0.0
                    for addr in usdc_addrs:
                        tok = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=erc20_abi)
                        raw = tok.functions.balanceOf(wallet_cs).call()
                        dec = tok.functions.decimals().call()
                        total += float(raw) / (10**dec)
                    return total
                except Exception:
                    continue  # try next RPC

            print("[LIVE] get_wallet_balance: all RPC endpoints failed")
            return None
        except Exception as e:
            print(f"[LIVE] get_wallet_balance failed: {e}")
            return None

    # Gnosis ConditionalTokens contract on Polygon — holds ERC-1155 shares and
    # exposes redeemPositions().  Polymarket's CTF Exchange wraps this contract.
    _CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    # NegRisk CTF contract for multi-outcome / negRisk markets
    _NEG_RISK_CTF_CONTRACT = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    # USDC.e collateral address on Polygon
    _USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

    _REDEEM_ABI = [
        {
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"},
            ],
            "name": "redeemPositions",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        }
    ]

    def redeem_winning_position(self, trade: "Trade", market: "Market") -> bool:
        """Call redeemPositions on-chain to collect USDC.e from a won position.

        Polymarket does NOT auto-redeem for raw EOA wallets — only for proxy-wallet
        accounts created through their web UI.  This method must be called manually
        after each win to recover the $1-per-share payout.

        UP wins   → outcome index 0 → indexSet = 1  (binary 01)
        DOWN wins → outcome index 1 → indexSet = 2  (binary 10)
        """
        if not trade.won:
            return False
        if not market.condition_id:
            print(f"[LIVE] redeem: no conditionId for market {market.slug} — skipping")
            return False

        try:
            from web3 import Web3

            ctf_addr = self._NEG_RISK_CTF_CONTRACT if market.neg_risk else self._CTF_CONTRACT
            # UP = outcome 0 → indexSet 1; DOWN = outcome 1 → indexSet 2
            index_set = 1 if trade.direction == "up" else 2
            condition_bytes = bytes.fromhex(market.condition_id.replace("0x", ""))

            candidates = [Config.POLYGON_RPC_URL] + [r for r in self._FALLBACK_RPCS if r != Config.POLYGON_RPC_URL]

            for rpc_url in candidates:
                try:
                    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
                    account = w3.eth.account.from_key(Config.PRIVATE_KEY)
                    ctf = w3.eth.contract(
                        address=Web3.to_checksum_address(ctf_addr),
                        abi=self._REDEEM_ABI,
                    )
                    tx = ctf.functions.redeemPositions(
                        Web3.to_checksum_address(self._USDC_E),
                        b"\x00" * 32,  # parentCollectionId = bytes32(0)
                        condition_bytes,
                        [index_set],
                    ).build_transaction(
                        {
                            "from": account.address,
                            "nonce": w3.eth.get_transaction_count(account.address),
                            "gas": 200_000,
                            "gasPrice": w3.eth.gas_price,
                        }
                    )
                    signed = account.sign_transaction(tx)
                    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                    if receipt["status"] == 1:
                        print(
                            f"[LIVE] Redeemed {trade.direction.upper()} win "
                            f"({trade.shares_bought:.4f} shares) "
                            f"tx={tx_hash.hex()[:16]}..."
                        )
                        return True
                    else:
                        print(f"[LIVE] Redeem tx reverted: {tx_hash.hex()[:16]}...")
                        return False
                except Exception:
                    continue

            print("[LIVE] redeem_winning_position: all RPC endpoints failed")
            return False
        except Exception as e:
            print(f"[LIVE] redeem_winning_position failed: {e}")
            return False

    def cancel_all_open_orders(self) -> int:
        """Cancel all open orders on CLOB (call at startup for crash recovery).

        Returns the number of orders cancelled.
        """
        try:
            all_orders = self.client.get_orders()
            # get_orders() returns a dict with a "data" key, or a list directly
            raw = all_orders.get("data", all_orders) if isinstance(all_orders, dict) else all_orders
            orders = [o for o in (raw or []) if isinstance(o, dict) and o.get("status", "").upper() == "OPEN"]
            for order in orders:
                try:
                    self.client.cancel(order["id"])
                except Exception as e:
                    print(f"[LIVE] Startup: failed to cancel order {order.get('id')}: {e}")
            if orders:
                print(f"[LIVE] Startup: cancelled {len(orders)} orphaned open order(s)")
            return len(orders)
        except Exception as e:
            print(f"[LIVE] Startup cleanup error: {e}")
            return 0
