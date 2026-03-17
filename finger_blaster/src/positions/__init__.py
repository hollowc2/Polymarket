"""
Position Manager - Monitor and manage wallet positions on Polymarket.

Provides:
- Real-time position monitoring with PnL tracking
- Filter for current 15m BTC market
- Position closing with confirmation
- WebSocket + polling for instant updates

Usage (standalone):
    python -m src.positions
    python main.py --positions

Usage (integrated):
    from src.positions import PositionsCore, PositionsConfig

    positions = PositionsCore(config=PositionsConfig())
    positions.on('positions_update', my_handler)

    await positions.start()
    # ... run your application ...
    await positions.stop()
"""

from src.positions.config import PositionsConfig
from src.positions.core import PositionsCore

__all__ = [
    "PositionsCore",
    "PositionsConfig",
]

__version__ = "0.1.0"
