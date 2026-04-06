from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class TradeRecord:
    """Unified trade record — normalises both backtest and live/paper schemas."""

    # identity
    id: str
    strategy: str
    source: Literal["backtest", "live"]

    # timing (always UTC datetime)
    open_time: datetime         # aligns to OHLCV candle open_time index
    executed_at: datetime | None  # actual execution moment (live only)

    # trade
    direction: Literal["up", "down"]
    amount: float               # USD size
    entry_price: float          # market price when signal fired
    fill_price: float | None    # actual fill price (live); None for backtest

    # outcome
    won: bool | None            # None = pending settlement
    pnl: float                  # net profit/loss in USD

    # live-only fields (all None for backtest)
    gate_name: str | None = None
    gate_boosted: bool | None = None
    slippage_pct: float | None = None
    spread: float | None = None
    fill_pct: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    price_movement_pct: float | None = None
    session_trade_n: int | None = None
    hour_utc: int | None = None
    day_of_week: int | None = None
    consecutive_wins: int | None = None
    consecutive_losses: int | None = None
    bankroll_before: float | None = None
    market_bias: str | None = None
    is_paper: bool | None = None
    market_slug: str | None = None
