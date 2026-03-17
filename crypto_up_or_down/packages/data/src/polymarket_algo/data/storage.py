from pathlib import Path

import pandas as pd


def write_candles(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)


def read_candles(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)
