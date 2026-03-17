"""
DeltaFlipStrategy — Zero-crossing detector on Hyperliquid cumulative delta.

**Live-only strategy.** Fires a directional signal the first cycle after
`cumulative_delta` (buy_vol - sell_vol) crosses zero, indicating a regime change
from net-buying to net-selling (or vice versa). This fires earlier than strategies
that wait for the 55/45 dominant_side threshold.

Signal logic:
    1. Load previous cumulative_delta from a JSON sidecar state file.
    2. Fetch current cumulative_delta from HL for `coin` at `timeframe`.
    3. If sign changed (prev * cur < 0) → signal in direction of NEW regime.
    4. HTF gate (default 4h): if 4h dominant_side opposes the flip → veto.
    5. Size = min(base_size, max_size) — flat sizing (binary flip signal).
    6. Save current delta to state file for next cycle.

State file: state/delta_flip_{COIN}_{TF}.json → {"prev_delta": 1234.56, "ts": 1710000000}
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
from polymarket_algo.indicators.hl_orderflow import hl_orderflow


def _state_path(state_dir: str, coin: str, tf: str) -> Path:
    return Path(state_dir) / f"delta_flip_{coin}_{tf}.json"


def _load_state(state_dir: str, coin: str, tf: str) -> dict:
    path = _state_path(state_dir, coin, tf)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state_dir: str, coin: str, tf: str, delta: float) -> None:
    path = _state_path(state_dir, coin, tf)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"prev_delta": delta, "ts": int(time.time())})
    # Atomic write via temp file + rename
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".delta_flip_")
    try:
        os.write(fd, data.encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


class DeltaFlipStrategy:
    name = "delta_flip"
    description = "Zero-crossing detector on Hyperliquid cumulative delta (live only)"
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            "coin": "BTC",
            "timeframe": "5m",
            "gate_timeframe": "4h",
            "base_size": 1.0,
            "max_size": 5.0,
            "state_dir": "state",
        }

    @property
    def param_grid(self) -> dict:
        return {
            "coin": ["BTC", "ETH", "SOL"],
            "timeframe": ["5m", "15m"],
            "base_size": [1.0, 2.0, 5.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        config = {**self.default_params, **params}

        coin = config["coin"]
        tf = config["timeframe"]
        state_dir = config["state_dir"]

        flow = hl_orderflow(coin)

        # Current cumulative delta at signal timeframe
        cur_delta: float = flow.get(tf, {}).get("cumulative_delta", 0.0)

        # HTF gate
        gate_side: str = flow.get(config["gate_timeframe"], {}).get("dominant_side", "NEUTRAL")

        # Zero-crossing detection
        state = _load_state(state_dir, coin, tf)
        prev_delta = state.get("prev_delta")  # None on first run

        if prev_delta is None:
            raw_signal = 0  # establish baseline — no signal first cycle
        elif prev_delta * cur_delta < 0:
            # Sign changed → flip detected; direction = new regime
            raw_signal = 1 if cur_delta > 0 else -1
        else:
            raw_signal = 0

        # HTF gate veto
        if raw_signal == 1 and gate_side == "SELL":
            raw_signal = 0
        elif raw_signal == -1 and gate_side == "BUY":
            raw_signal = 0

        # Flat sizing (binary signal)
        size = min(config["base_size"], config["max_size"]) if raw_signal != 0 else 0.0

        # Persist state for next cycle
        _save_state(state_dir, coin, tf, cur_delta)

        return pd.DataFrame({"signal": raw_signal, "size": size}, index=candles.index)
