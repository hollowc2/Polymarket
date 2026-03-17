import pandas as pd
from polymarket_algo.indicators import ema, macd, rsi, sma


def test_ema_sma_shapes() -> None:
    s = pd.Series(range(1, 50))
    assert len(ema(s, 10)) == len(s)
    assert len(sma(s, 10)) == len(s)


def test_rsi_range_and_shape() -> None:
    s = pd.Series([100] * 30)
    out = rsi(s, 14)
    assert len(out) == len(s)
    assert out.dropna().between(0, 100).all()


def test_macd_dataframe_columns() -> None:
    s = pd.Series(range(1, 100))
    out = macd(s)
    assert set(out.columns) == {"macd", "signal", "histogram"}
    assert len(out) == len(s)
