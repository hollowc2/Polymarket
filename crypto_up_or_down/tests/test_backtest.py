import pandas as pd
from polymarket_algo.backtest.engine import run_backtest


def always_up(candles: pd.DataFrame, **_) -> pd.DataFrame:
    return pd.DataFrame({"signal": [1] * len(candles), "size": [10.0] * len(candles)}, index=candles.index)


def test_backtest_runs_on_synthetic_data() -> None:
    idx = pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC")
    closes = pd.Series(range(100, 150), index=idx)
    candles = pd.DataFrame({"close": closes}, index=idx)
    result = run_backtest(candles, always_up)
    assert "win_rate" in result.metrics
    assert result.metrics["trade_count"] > 0
