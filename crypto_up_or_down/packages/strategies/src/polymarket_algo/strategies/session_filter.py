from __future__ import annotations

import pandas as pd


class SessionFilter:
    """Composable post-process filter that zeroes out signals outside allowed UTC hours.

    Usage:
        signals = strategy.evaluate(candles)
        sf = SessionFilter(allowed_hours=[(13, 20)])  # US session only
        signals = sf.apply(signals, candles)
    """

    def __init__(self, allowed_hours: list[tuple[int, int]] | None = None) -> None:
        # allowed_hours: list of (start_hour, end_hour) UTC ranges, inclusive.
        # None or empty list = no filtering (pass-through).
        self.allowed_hours = allowed_hours or []

    def apply(self, signals: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        """Zero out signal column for candles outside allowed UTC hours."""
        if not self.allowed_hours:
            return signals

        hour = candles.index.hour  # type: ignore[union-attr]  # UTC
        in_session = pd.Series(False, index=candles.index)
        for start, end in self.allowed_hours:
            in_session |= (hour >= start) & (hour <= end)

        out = signals.copy()
        out.loc[~in_session, "signal"] = 0
        # Also zero size for filtered-out signals
        if "size" in out.columns:
            out.loc[~in_session, "size"] = 0.0
        return out

    @classmethod
    def from_config(cls, config: object) -> SessionFilter:
        """Parse SESSION_FILTER_HOURS string like '13-20,22-23' into allowed_hours."""
        raw: str = getattr(config, "SESSION_FILTER_HOURS", "") or ""
        raw = raw.strip()
        if not raw:
            return cls(allowed_hours=None)

        allowed: list[tuple[int, int]] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                allowed.append((int(start_str.strip()), int(end_str.strip())))
            else:
                h = int(part)
                allowed.append((h, h))

        return cls(allowed_hours=allowed or None)
