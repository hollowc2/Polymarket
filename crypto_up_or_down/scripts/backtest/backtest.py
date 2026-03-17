from pathlib import Path

import pandas as pd
from polymarket_algo.backtest.engine import parameter_sweep, run_backtest, walk_forward_split
from polymarket_algo.strategies.candle_direction import CandleDirectionStrategy

PARAM_GRID = {
    "ema_fast": [8, 12],
    "ema_slow": [21, 26],
    "rsi_period": [10, 14],
    "rsi_overbought": [65, 70],
    "rsi_oversold": [30, 35],
    "macd_fast": [8, 12],
    "macd_slow": [21, 26],
    "macd_signal": [9],
}


def main() -> None:
    df = pd.read_parquet(Path("data") / "btc_1h.parquet")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    candles = df.set_index("open_time").sort_index()
    train, test = walk_forward_split(candles)
    strategy = CandleDirectionStrategy()
    sweep = parameter_sweep(train, strategy.evaluate, PARAM_GRID)
    best = sweep.iloc[0].to_dict()
    params = {k: best[k] for k in PARAM_GRID}
    result = run_backtest(test, strategy.evaluate, params)
    print(result.metrics)


if __name__ == "__main__":
    main()
