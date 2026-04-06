from __future__ import annotations

from typing import Any

import pandas as pd
from polymarket_algo.indicators.hl_orderflow import hl_orderflow_signal


class ThreeBarMoMoStrategy:
    name = "3bar_momo"
    description = "Momentum: N consecutive bars same direction with strictly increasing volume"
    timeframe = "5m"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "bars": 3,  # consecutive qualifying bars required
            "size": 15.0,  # base bet in USD
            "size_cap": 2.0,  # max multiplier for vol-scaled size
            "min_body_pct": 0.0,  # min candle body as % of close (0 = off)
            "max_vol_spike": 3.0,  # veto if current vol > N×20-bar mean (0 = off)
            "vol_spike_lookback": 20,  # rolling window for vol spike baseline
            "hl_gate": False,  # veto signals when HL 5m+15m both oppose
            "hl_coin": "BTC",  # coin to check for HL gate
        }

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "bars": [2, 3, 4, 5],
            "size": [10.0, 15.0, 20.0],
            "size_cap": [1.5, 2.0, 3.0],
            "min_body_pct": [0.0, 0.001, 0.002, 0.005],
            "max_vol_spike": [0.0, 2.5, 3.0, 4.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params: Any) -> pd.DataFrame:
        config = {**self.default_params, **params}
        bars = int(config["bars"])
        size_val = float(config["size"])
        size_cap = float(config["size_cap"])
        min_body_pct = float(config["min_body_pct"])
        max_vol_spike = float(config["max_vol_spike"])
        vol_spike_lookback = int(config["vol_spike_lookback"])

        body = candles["close"] - candles["open"]
        # Vectorized direction: 1 (bullish), -1 (bearish), 0 (doji)
        direction = (body > 0).astype(int) - (body < 0).astype(int)
        body_pct = body.abs() / candles["close"]
        volumes = candles["volume"]

        # All N bars same direction (non-zero)
        all_bullish = (direction == 1).rolling(bars, min_periods=bars).min().fillna(0).astype(bool)
        all_bearish = (direction == -1).rolling(bars, min_periods=bars).min().fillna(0).astype(bool)

        # Strictly increasing volume: bars-1 consecutive positive diffs
        if bars > 1:
            all_vol_increasing = (
                (volumes.diff() > 0).rolling(bars - 1, min_periods=bars - 1).min().fillna(0).astype(bool)
            )
        else:
            all_vol_increasing = pd.Series(True, index=candles.index)

        # Optional minimum body size filter
        if min_body_pct > 0:
            body_ok = (body_pct >= min_body_pct).rolling(bars, min_periods=bars).min().fillna(0).astype(bool)
        else:
            body_ok = pd.Series(True, index=candles.index)

        # Volume ratio: last bar / first bar in the window, capped
        vol_first = volumes.shift(bars - 1)
        vol_ratio = (volumes / vol_first).clip(upper=size_cap).fillna(1.0)

        # Vol spike gate: veto signals when current bar volume is an outlier vs rolling baseline
        if max_vol_spike > 0:
            vol_ma = volumes.rolling(vol_spike_lookback, min_periods=vol_spike_lookback // 2).mean()
            vol_spike_ok = (volumes / vol_ma.replace(0, float("nan"))) <= max_vol_spike
            vol_spike_ok = vol_spike_ok.fillna(True)
        else:
            vol_spike_ok = pd.Series(True, index=candles.index)

        bullish = all_bullish & all_vol_increasing & body_ok & vol_spike_ok
        bearish = all_bearish & all_vol_increasing & body_ok & vol_spike_ok

        signal = bullish.astype(int) - bearish.astype(int)
        size = pd.Series(0.0, index=candles.index)
        size[signal != 0] = size_val * vol_ratio[signal != 0]

        out = pd.DataFrame({"signal": signal, "size": size}, index=candles.index)

        if config.get("hl_gate"):
            coin = str(config.get("hl_coin", "BTC"))
            sig_5m = hl_orderflow_signal(coin, "5m")
            sig_15m = hl_orderflow_signal(coin, "15m")
            # Only veto when BOTH timeframes strongly oppose the signal direction
            if sig_5m == "SELL" and sig_15m == "SELL":
                out.loc[out["signal"] == 1, ["signal", "size"]] = [0, 0.0]
            if sig_5m == "BUY" and sig_15m == "BUY":
                out.loc[out["signal"] == -1, ["signal", "size"]] = [0, 0.0]

        return out
