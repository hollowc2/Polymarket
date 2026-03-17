import pandas as pd
from polymarket_algo.core.types import Strategy
from polymarket_algo.strategies.streak_reversal import StreakReversalStrategy


def test_streak_strategy_conforms_protocol() -> None:
    strategy: Strategy = StreakReversalStrategy()
    idx = pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC")
    candles = pd.DataFrame({"close": range(20)}, index=idx)
    out = strategy.evaluate(candles)
    assert {"signal", "size"}.issubset(out.columns)
