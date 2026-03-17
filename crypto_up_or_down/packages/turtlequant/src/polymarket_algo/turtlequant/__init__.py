"""TurtleQuant — Probabilistic pricing engine for Polymarket crypto option markets.

Exploits mispricings on longer-term Polymarket prediction markets (e.g.,
"Will BTC be above $75k by March 30?") by computing P(S_T > K) using
professional derivatives math and trading where the gap exceeds a threshold.

Components:
    market_scanner   — Gamma API discovery + liquidity/spread/time filters
    market_parser    — regex + rule templates → (asset, strike, expiry, option_type)
    vol_surface      — Deribit IV fetcher + strike/expiry interpolation + realized vol fallback
    probability_engine — digital_probability(), barrier_probability()
    position_manager — open positions, Kelly sizing, exit logic, NAV limits
"""

from .market_parser import MarketParams, OptionType, parse_market
from .market_scanner import ActiveMarket, MarketScanner
from .position_manager import PositionManager
from .probability_engine import (
    barrier_down_probability,
    barrier_probability,
    compute_probability,
    digital_probability,
    european_put_probability,
)
from .vol_surface import VolSurface

__all__ = [
    "ActiveMarket",
    "MarketParams",
    "MarketScanner",
    "OptionType",
    "PositionManager",
    "VolSurface",
    "barrier_down_probability",
    "barrier_probability",
    "compute_probability",
    "digital_probability",
    "european_put_probability",
    "parse_market",
]
