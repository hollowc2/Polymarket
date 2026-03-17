"""Adapters bridging live market data to Strategy Protocol input and back."""

import pandas as pd

from .sizing import BetDecision, get_reversal_rate, kelly_size

# How many raw 5m Polymarket outcomes make up one bar at each timeframe.
TF_GROUP_SIZE: dict[str, int] = {"5m": 1, "15m": 3, "1h": 12}


def detect_streak(outcomes: list[str]) -> tuple[int, str]:
    """Count the trailing streak from a list of outcomes.

    Args:
        outcomes: e.g. ["up", "up", "down", "up", "up", "up"]

    Returns:
        (streak_length, streak_direction) — e.g. (3, "up")
    """
    if not outcomes:
        return 0, ""

    current = outcomes[-1]
    streak = 1

    for i in range(len(outcomes) - 2, -1, -1):
        if outcomes[i] == current:
            streak += 1
        else:
            break

    return streak, current


def resample_outcomes(outcomes: list[str], group_size: int) -> list[str]:
    """Aggregate consecutive 5m outcomes into larger-timeframe bars.

    Each group of `group_size` outcomes becomes one bar: "up" if the majority
    are "up", "down" otherwise (ties go to "down" — conservative).

    Args:
        outcomes: Raw 5m outcome strings, oldest first.
        group_size: Number of 5m outcomes per bar (1=5m, 3=15m, 12=1h).

    Returns:
        Resampled outcome list, length = len(outcomes) // group_size.
    """
    if group_size == 1:
        return outcomes
    resampled = []
    n = (len(outcomes) // group_size) * group_size
    for i in range(0, n, group_size):
        group = outcomes[i : i + group_size]
        ups = sum(1 for o in group if o == "up")
        resampled.append("up" if ups > group_size / 2 else "down")
    return resampled


def outcomes_to_candles(outcomes: list[str]) -> pd.DataFrame:
    """Convert outcome strings to a candle DataFrame suitable for StreakReversalStrategy.

    Uses cumulative values so that consecutive same-direction outcomes produce
    a positive diff (close.diff() > 0 for "up", < 0 for "down"). Binary 1/0
    would break because consecutive same-direction diffs would be 0.

    Args:
        outcomes: e.g. ["up", "up", "down", "up"] (oldest first)

    Returns:
        DataFrame with open, high, low, close columns (all same cumulative value)
    """
    # Start with a seed value of 0 so the first outcome produces a meaningful
    # diff (positive for "up", negative for "down") instead of NaN.
    cumulative = 0.0
    values = [cumulative]  # seed row
    for o in outcomes:
        if o == "up":
            cumulative += 1.0
        else:
            cumulative -= 1.0
        values.append(cumulative)

    df = pd.DataFrame({"open": values, "high": values, "low": values, "close": values})
    return df


def interpret_signal(
    result: pd.DataFrame,
    outcomes: list[str],
    bankroll: float,
    entry_price: float,
    max_bet: float,
    max_bankroll_pct: float = 0.1,
    timeframe: str = "5m",
    asset: str = "",
) -> BetDecision:
    """Read last row of strategy output and produce a BetDecision.

    Args:
        result: DataFrame from strategy.evaluate() with "signal" and "size" columns
        outcomes: Resampled outcomes at the target timeframe (for streak detection).
        bankroll: Current bankroll in USD
        entry_price: Market price for the bet direction
        max_bet: Maximum bet amount from CLI/config
        max_bankroll_pct: Never risk more than this fraction of bankroll
        timeframe: Candle timeframe used — selects the correct REVERSAL_RATES table.
        asset: Asset ticker (e.g. "BTC", "SOL") for asset-specific rates; "" → BTC table.

    Returns:
        BetDecision ready for trader.place_bet()
    """
    last = result.iloc[-1]
    signal = int(last["signal"])

    if signal == 0:
        return BetDecision(
            should_bet=False,
            direction="",
            size=0,
            confidence=0,
            reason="No signal from strategy",
        )

    direction = "down" if signal == -1 else "up"

    # Look up confidence from the timeframe-specific reversal rate table
    streak_len, streak_dir = detect_streak(outcomes)
    confidence = get_reversal_rate(timeframe, streak_len, asset)

    # Calculate Kelly-optimal size
    odds = 1.0 / entry_price if entry_price > 0 else 2.0
    kelly = kelly_size(confidence, odds, bankroll)

    size = min(kelly, max_bet, bankroll * max_bankroll_pct)
    size = max(1.0, size)  # floor at $1

    bet_direction_label = direction.upper()
    reason = (
        f"Streak of {streak_len}x {streak_dir} ({timeframe}) detected. "
        f"Reversal rate: {confidence:.1%}. "
        f"Betting {bet_direction_label} (Kelly=${kelly:.2f}, capped=${size:.2f})."
    )

    return BetDecision(
        should_bet=True,
        direction=direction,
        size=size,
        confidence=confidence,
        reason=reason,
    )
