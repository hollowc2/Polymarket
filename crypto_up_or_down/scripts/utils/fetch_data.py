from datetime import UTC, datetime
from pathlib import Path

from polymarket_algo.data.binance import INTERVALS, START, SYMBOLS, fetch_klines


def main() -> None:
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    for symbol in SYMBOLS:
        asset = symbol.replace("USDT", "").lower()
        for interval in INTERVALS:
            df = fetch_klines(symbol, interval, start_ms, end_ms)
            out = data_dir / f"{asset}_{interval}.parquet"
            df.to_parquet(out, index=False)
            print(f"Saved {len(df):,} candles -> {out}")


if __name__ == "__main__":
    main()
