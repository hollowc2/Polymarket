"""
HLOrderFlowReversalStrategy — Contrarian exhaustion signal from Hyperliquid perp order flow.

**Live-only strategy.** Each call to `evaluate()` fetches the current Hyperliquid
order flow snapshot via `hl_orderflow()`. There is no historical HL data available,
so backtesting this strategy will always reflect *current* market conditions, not
the conditions that existed when the candles were recorded. Use backtests only for
plumbing/wiring checks, not for historical performance measurement.

Signal logic (inverted from HLOrderFlowMomentumStrategy):
    1. Fetch HL order flow for `coin` across all timeframes.
    2. HTF gate (default 4h) — INVERTED: if 4h dominant_side == BUY, veto SELL signals
       (don't fade in a strong uptrend); if 4h == SELL, veto BUY signals (don't fade
       in a strong downtrend). Only fade when the HTF is neutral or counter to the flow.
    3. Entry vote across 5m, 15m, 1h (default): need >= min_votes (default 3) on same
       side. Stricter than momentum (2-of-3) to reduce false exhaustion signals.
    4. Raw signal is INVERTED: heavy HL buying → -1 (fade DOWN); heavy selling → +1 (fade UP).
    5. Size = base_size × exhausted_pressure × 2, capped at max_size.
       Pressure source: buy_pressure when fading buyers; sell_pressure when fading sellers.
"""

import pandas as pd
from polymarket_algo.indicators.hl_orderflow import hl_orderflow


class HLOrderFlowReversalStrategy:
    name = "hl_orderflow_reversal"
    description = "Contrarian exhaustion signal from Hyperliquid perp order flow (live only)"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "coin": "BTC",
            "size": 15.0,
            "max_size": 30.0,
            "min_votes": 3,
            "entry_timeframes": ["5m", "15m", "1h"],
            "gate_timeframe": "4h",
            "pressure_tf": "15m",
        }

    @property
    def param_grid(self) -> dict:
        return {
            "coin": ["BTC", "ETH", "SOL"],
            "size": [10.0, 15.0, 20.0],
            "min_votes": [2, 3],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        config = {**self.default_params, **params}

        flow = hl_orderflow(config["coin"])

        # HTF gate — INVERTED vs momentum:
        #   BUY  gate → veto -1 (don't fade buyers in strong uptrend)
        #   SELL gate → veto +1 (don't fade sellers in strong downtrend)
        gate_side = flow.get(config["gate_timeframe"], {}).get("dominant_side", "NEUTRAL")

        # Entry vote
        buy_votes = 0
        sell_votes = 0
        for tf in config["entry_timeframes"]:
            side = flow.get(tf, {}).get("dominant_side", "NEUTRAL")
            if side == "BUY":
                buy_votes += 1
            if side == "SELL":
                sell_votes += 1

        min_v = config["min_votes"]
        # INVERTED: heavy HL buys → fade DOWN; gate blocks when 4h also bullish
        if buy_votes >= min_v and gate_side != "BUY":
            raw_signal = -1
        elif sell_votes >= min_v and gate_side != "SELL":
            raw_signal = 1
        else:
            raw_signal = 0

        # Sizing via pressure of the exhausted side
        pressure_data = flow.get(config["pressure_tf"], {})
        if raw_signal == -1:
            pressure = pressure_data.get("buy_pressure", 0.5)
        elif raw_signal == 1:
            pressure = pressure_data.get("sell_pressure", 0.5)
        else:
            pressure = 0.0

        size = min(config["size"] * pressure * 2, config["max_size"]) if raw_signal != 0 else 0.0

        return pd.DataFrame({"signal": raw_signal, "size": size}, index=candles.index)
