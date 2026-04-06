from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from polymarket_algo.data.binance import INTERVALS, START, SYMBOLS, fetch_klines


def main() -> None:
    end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    for symbol in SYMBOLS:
        asset = symbol.replace("USDT", "").lower()
        for interval in INTERVALS:
            out = data_dir / f"{asset}_{interval}.parquet"

            # Incremental: start from the last stored candle if the file exists
            if out.exists():
                existing = pd.read_parquet(out)
                existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
                last_ts = existing["open_time"].max()
                start_ms = int(last_ts.timestamp() * 1000) + 1
                print(f"[{asset}/{interval}] incremental from {last_ts.isoformat()}")
            else:
                start_ms = int(START.timestamp() * 1000)
                existing = None
                print(f"[{asset}/{interval}] full fetch from {START.isoformat()}")

            new_df = fetch_klines(symbol, interval, start_ms, end_ms)

            if new_df.empty:
                print(f"[{asset}/{interval}] nothing new")
                continue

            if existing is not None:
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True)
                combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
            else:
                combined = new_df

            combined.to_parquet(out, index=False)
            print(f"[{asset}/{interval}] saved {len(new_df):,} new candles → {len(combined):,} total → {out}")


if __name__ == "__main__":
    main()
