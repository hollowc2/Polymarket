"""Configuration for Position Manager module."""

from dataclasses import dataclass, field
from typing import Set


@dataclass
class PositionsConfig:
    """Configuration for PositionsCore."""

    # Polling intervals
    position_poll_interval: float = 2.0  # Seconds between position API polls (reduced for faster updates)
    market_poll_interval: float = 5.0  # Seconds between market discovery polls

    # API settings
    request_timeout: float = 10.0
    min_position_size: float = 0.1  # Minimum position size to display

    # Market filter
    btc_series_id: str = "10192"  # BTC Up or Down 15m series

    # Display settings
    show_closed_positions: bool = False  # Show positions with 0 size
