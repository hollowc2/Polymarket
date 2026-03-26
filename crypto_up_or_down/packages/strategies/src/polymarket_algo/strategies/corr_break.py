"""BTC/ETH Correlation Break strategy.

BTC and ETH normally exhibit ~0.85-0.95 rolling correlation. When this correlation
breaks down (one asset diverges from the other), the laggard historically catches
up — or the leader reverts.

Signal logic:
  1. Compute rolling correlation of BTC and ETH log-returns.
  2. When correlation drops below corr_thresh → correlation break detected.
  3. Compute BTC/ETH spread z-score (rolling).
  4. If primary asset is BTC:
       spread z < -thresh (ETH expensive vs BTC) → BTC should catch up → signal UP
       spread z > +thresh (BTC expensive vs ETH) → BTC should revert → signal DOWN
  5. If primary asset is ETH: invert logic.

The strategy fetches the secondary asset's candles internally (same time range).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CorrBreakStrategy:
    """BTC/ETH correlation break: bet on laggard catching up or leader reverting."""

    name = "corr_break"
    description = "Trade cross-asset correlation breaks between BTC and ETH"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "corr_window": 20,        # bars for rolling correlation
            "corr_thresh": 0.7,       # below this → correlation break
            "spread_z_thresh": 1.5,   # spread z-score threshold for signal
            "spread_window": 20,      # bars for spread z-score
            "primary_asset": "btc",   # asset whose market we're trading
            "size": 15.0,
        }

    @property
    def param_grid(self) -> dict:
        return {
            "corr_window": [15, 20, 30],
            "corr_thresh": [0.5, 0.7, 0.8],
            "spread_z_thresh": [1.0, 1.5, 2.0],
            "size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate BTC/ETH correlation break signals.

        Args:
            candles: OHLCV DataFrame for the PRIMARY asset, indexed by open_time.
                     The strategy fetches secondary asset candles internally.
            **params: corr_window, corr_thresh, spread_z_thresh, spread_window,
                      primary_asset, size.
                      Optionally pass 'secondary_candles' (pd.DataFrame) to avoid
                      a network fetch (useful for backtesting efficiency).

        Returns:
            DataFrame with 'signal' (1/-1/0) and 'size' columns.
        """
        corr_window = int(params.get("corr_window", self.default_params["corr_window"]))
        corr_thresh = float(params.get("corr_thresh", self.default_params["corr_thresh"]))
        spread_z_thresh = float(params.get("spread_z_thresh", self.default_params["spread_z_thresh"]))
        spread_window = int(params.get("spread_window", self.default_params["spread_window"]))
        primary_asset = str(params.get("primary_asset", self.default_params["primary_asset"])).lower()
        size_val = float(params.get("size", self.default_params["size"]))

        no_signal = pd.DataFrame({"signal": 0, "size": 0.0}, index=candles.index)

        # Get secondary candles: passed in or fetched
        secondary_candles = params.get("secondary_candles")
        if secondary_candles is None:
            secondary_candles = self._fetch_secondary(candles, primary_asset)

        if secondary_candles is None or secondary_candles.empty:
            return no_signal

        # Align both close series on candle index
        sec_close = secondary_candles["close"].reindex(candles.index)

        if primary_asset == "btc":
            btc_close = candles["close"]
            eth_close = sec_close
        else:
            btc_close = sec_close
            eth_close = candles["close"]

        btc_ret = np.log(btc_close / btc_close.shift(1))
        eth_ret = np.log(eth_close / eth_close.shift(1))

        # Rolling correlation
        rolling_corr = btc_ret.rolling(corr_window, min_periods=corr_window // 2).corr(eth_ret)

        # Spread: log(BTC/ETH) z-scored
        spread = np.log(btc_close / eth_close.replace(0, float("nan")))
        roll_mean = spread.rolling(spread_window, min_periods=5).mean()
        roll_std = spread.rolling(spread_window, min_periods=5).std()
        spread_z = (spread - roll_mean) / roll_std.replace(0.0, float("nan"))
        spread_z = spread_z.fillna(0.0)

        signal = pd.Series(0, index=candles.index, dtype=int)
        corr_break = rolling_corr < corr_thresh

        if primary_asset == "btc":
            # BTC lagging (ETH expensive vs BTC): spread_z < -thresh → BTC UP
            signal[corr_break & (spread_z < -spread_z_thresh)] = 1
            # BTC leading (BTC expensive vs ETH): spread_z > +thresh → BTC DOWN
            signal[corr_break & (spread_z > spread_z_thresh)] = -1
        else:
            # ETH lagging (BTC expensive vs ETH): spread_z > +thresh → ETH UP
            signal[corr_break & (spread_z > spread_z_thresh)] = 1
            # ETH leading (ETH expensive vs BTC): spread_z < -thresh → ETH DOWN
            signal[corr_break & (spread_z < -spread_z_thresh)] = -1

        signal = signal.fillna(0).astype(int)
        size = pd.Series(size_val, index=candles.index)
        size[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)

    def _fetch_secondary(self, candles: pd.DataFrame, primary_asset: str) -> pd.DataFrame | None:
        """Fetch candles for the secondary asset (same time range as primary)."""
        try:
            from polymarket_algo.data.binance import fetch_klines

            secondary_symbol = "ETHUSDT" if primary_asset == "btc" else "BTCUSDT"
            start_ms = int(candles.index.min().timestamp() * 1000)
            end_ms = int(candles.index.max().timestamp() * 1000) + 1

            # Infer interval from spacing
            interval = "5m"
            if len(candles) >= 2:
                spacing_s = int((candles.index[1] - candles.index[0]).total_seconds())
                mapping = {300: "5m", 900: "15m", 3600: "1h", 14400: "4h"}
                interval = mapping.get(spacing_s, "5m")

            df = fetch_klines(secondary_symbol, interval, start_ms, end_ms)
            if df.empty:
                return None
            return df.set_index("open_time").sort_index()
        except Exception as exc:
            print(f"[corr_break] Secondary asset fetch failed: {exc}")
            return None
