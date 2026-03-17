"""Selective trade quality filter for copytrade signals."""

from __future__ import annotations

from polymarket_algo.core.config import Config


class SelectiveFilter:
    """Filters trades based on analysis of copytrade patterns."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}

        self.max_delay_ms = int(cfg.get("max_delay_ms", Config.SELECTIVE_MAX_DELAY_MS))
        self.min_fill_price = float(cfg.get("min_fill_price", Config.SELECTIVE_MIN_FILL_PRICE))
        self.max_fill_price = float(cfg.get("max_fill_price", Config.SELECTIVE_MAX_FILL_PRICE))
        self.max_price_movement_pct = float(cfg.get("max_price_movement_pct", Config.SELECTIVE_MAX_PRICE_MOVEMENT_PCT))
        self.max_spread = float(cfg.get("max_spread", Config.SELECTIVE_MAX_SPREAD))
        self.max_volatility_factor = float(cfg.get("max_volatility_factor", Config.SELECTIVE_MAX_VOLATILITY_FACTOR))
        self.min_depth_at_best = float(cfg.get("min_depth_at_best", Config.SELECTIVE_MIN_DEPTH_AT_BEST))

    def should_trade(self, signal, market, execution_info: dict) -> tuple[bool, str]:
        """Return (should_trade, reason_if_skipped)."""
        delay_ms = int(execution_info.get("copy_delay_ms", 0))
        fill_price = float(execution_info.get("execution_price", 0.0))
        spread = float(execution_info.get("spread", 0.0))
        price_movement_pct = abs(float(execution_info.get("price_movement_pct", 0.0)))
        depth_at_best = float(execution_info.get("depth_at_best", 0.0))

        delay_breakdown = execution_info.get("delay_breakdown") or {}
        volatility_factor = float(delay_breakdown.get("volatility_factor", 1.0))

        if delay_ms > self.max_delay_ms:
            return False, f"delay {delay_ms / 1000:.1f}s > {self.max_delay_ms / 1000:.1f}s max"

        if fill_price > 0 and fill_price < self.min_fill_price:
            return False, f"fill_price {fill_price:.2f} < {self.min_fill_price:.2f} min"

        if fill_price > self.max_fill_price:
            return False, f"fill_price {fill_price:.2f} > {self.max_fill_price:.2f} max"

        if price_movement_pct > self.max_price_movement_pct:
            return False, (f"price_move {price_movement_pct:.1f}% > {self.max_price_movement_pct:.1f}% max")

        if spread > self.max_spread:
            return False, f"spread {spread:.3f} > {self.max_spread:.3f} max"

        if volatility_factor >= self.max_volatility_factor:
            return False, (f"volatility_factor {volatility_factor:.2f} >= {self.max_volatility_factor:.2f} max")

        if depth_at_best < self.min_depth_at_best:
            return False, f"depth {depth_at_best:.2f} < {self.min_depth_at_best:.2f} min"

        return True, "all checks OK"
