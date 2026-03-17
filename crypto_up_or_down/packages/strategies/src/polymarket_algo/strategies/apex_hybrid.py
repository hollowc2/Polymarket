"""APEX Hybrid — Streak Reversal gated by Microstructure Exhaustion.

Combines the proven streak reversal edge with microstructure flow signals
used as *exhaustion* confirmation rather than momentum (which is what the
backtests showed they actually are at 5m resolution).

Empirical findings from walk-forward analysis (2022-2024 train / 2024+ test):
    TFI corr_forward  = -0.025   ← weak REVERSAL signal (buy flow reverts)
    OBI corr_forward  = -0.014   ← weak REVERSAL signal
    microprice_drift  = +0.028   ← slight MOMENTUM signal (overextension)
    hawkes_centred    = -0.0004  ← no directional content; always positive

Design
------
1. Streak detection (identical to StreakReversalStrategy)
       After ≥ trigger consecutive UP candles  → signal = -1 (bet DOWN)
       After ≥ trigger consecutive DOWN candles → signal = +1 (bet UP)

2. Exhaustion score — confirms the streak is "overdue" to reverse
       tfi_exhaust  = streak_dir × TFI_rolling_mean   (positive = buy/sell flow
                                                         drove the streak, reversal likely)
       mp_exhaust   = streak_dir × microprice_drift    (positive = microprice extended
                                                         in streak direction)
       exhaust      = w_tfi × clip(tfi_exhaust, 0) + w_mp × clip(mp_exhaust, 0)

   Only positive exhaustion counts — if flow contradicts the streak, we don't
   penalise the signal, we just use the base size.

3. Hawkes activity gate — skip dead markets
       Only trade when hawkes_intensity > hawkes_threshold
       (illiquid, quiet periods have poor fill quality and noisy signals)

4. Regime gate (from apex_filters)
       ATR in [25th, 85th] percentile and volume above 20th percentile

5. Kelly-fractional sizing boosted by exhaustion strength
       base_size = kelly_scale × max_bet
       size      = base_size × (1 + exhaust_mult × exhaust)   capped at max_bet

Requires
--------
enrich_candles() for full feature coverage (buy_vol, sell_vol, delta from
Binance taker_buy columns).  Falls back gracefully on plain OHLCV — streak
signal still fires, exhaustion boost is 0, Hawkes gate uses volume proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .apex_features import (
    compute_hawkes_intensity,
    compute_microprice_drift,
    compute_tfi,
)
from .apex_filters import detect_regime


class ApexHybridStrategy:
    """Streak reversal gated and sized by microstructure exhaustion signals."""

    name = "apex_hybrid"
    description = (
        "Streak reversal + flow exhaustion: TFI/microprice confirm overextension; Hawkes gates on market activity"
    )
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            # Streak
            "trigger": 4,
            # Exhaustion feature weights (0 = disabled)
            "w_tfi_exhaust": 1.0,
            "w_mp_exhaust": 0.5,
            # Exhaustion sizing multiplier
            "exhaust_mult": 1.0,
            # Hawkes gate: minimum intensity to trade (0 = disabled)
            "hawkes_threshold": 0.53,
            # Regime gate
            "use_regime": True,
            # Sizing
            "kelly_scale": 0.25,
            "max_bet": 15.0,
        }

    @property
    def param_grid(self) -> dict[str, list]:
        return {
            "trigger": [3, 4, 5],
            "w_tfi_exhaust": [0.0, 0.5, 1.0, 2.0],
            "w_mp_exhaust": [0.0, 0.5, 1.0],
            "exhaust_mult": [0.5, 1.0, 2.0],
            "hawkes_threshold": [0.0, 0.52, 0.55],
            "kelly_scale": [0.20, 0.30],
            "max_bet": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate APEX Hybrid strategy.

        Args:
            candles: OHLCV DataFrame indexed by open_time (UTC).
                     Enriched with buy_vol/sell_vol/delta for full feature
                     coverage; degrades gracefully on plain OHLCV.
            **params: Any key from default_params / param_grid.

        Returns:
            DataFrame with columns:
                signal   int    1 (UP), -1 (DOWN), 0 (no trade)
                size     float  USD bet size (0 when signal == 0)

            Diagnostic columns included for sweep inspection:
                streak, streak_dir, tfi_exhaust, mp_exhaust,
                exhaust_score, hawkes_intensity, regime_ok
        """
        p = {**self.default_params, **params}

        trigger = int(p["trigger"])
        w_tfi = float(p["w_tfi_exhaust"])
        w_mp = float(p["w_mp_exhaust"])
        exhaust_mult = float(p["exhaust_mult"])
        hawkes_threshold = float(p["hawkes_threshold"])
        use_regime = bool(p["use_regime"])
        kelly_scale = float(p["kelly_scale"])
        max_bet = float(p["max_bet"])

        # ── 1. Streak detection ───────────────────────────────────────
        direction = (candles["close"].diff() > 0).map({True: 1, False: -1}).fillna(0)
        streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1

        # streak_signal: what the streak says to do
        streak_signal = pd.Series(0, index=candles.index, dtype=int)
        streak_signal[(streak >= trigger) & (direction == 1)] = -1  # UP streak → bet DOWN
        streak_signal[(streak >= trigger) & (direction == -1)] = 1  # DOWN streak → bet UP

        # streak_dir: +1 if UP streak is running, -1 if DOWN streak
        # Used to orient exhaustion scores relative to streak direction
        streak_dir = -streak_signal  # +1 for UP streak, -1 for DOWN streak

        # ── 2. Exhaustion features ────────────────────────────────────
        # TFI: rolling mean over trigger bars
        # Exhaustion: streak_dir × TFI_mean > 0 means flow drove the streak
        tfi = compute_tfi(candles)
        tfi_mean = tfi.rolling(trigger, min_periods=1).mean()
        tfi_exhaust = (streak_dir * tfi_mean).clip(lower=0.0)

        # Microprice drift: current overextension in streak direction
        mpd = compute_microprice_drift(candles)
        mp_exhaust = (streak_dir * mpd).clip(lower=0.0)

        # Combined exhaustion (only positive contribution counted)
        exhaust_score = w_tfi * tfi_exhaust + w_mp * mp_exhaust

        # ── 3. Hawkes activity gate ───────────────────────────────────
        hawkes = compute_hawkes_intensity(candles)
        if hawkes_threshold > 0:
            hawkes_ok = hawkes >= hawkes_threshold
        else:
            hawkes_ok = pd.Series(True, index=candles.index)

        # ── 4. Regime gate ────────────────────────────────────────────
        if use_regime:
            regime_ok = detect_regime(candles)
        else:
            regime_ok = pd.Series(True, index=candles.index)

        # ── 5. Signal: streak + all gates ────────────────────────────
        gate_ok = hawkes_ok.values & regime_ok.values
        signal_vals = np.where(gate_ok, streak_signal.values, 0)
        signal = pd.Series(signal_vals.astype(int), index=candles.index)

        # ── 6. Kelly sizing boosted by exhaustion ─────────────────────
        base_size = kelly_scale * max_bet
        boost = 1.0 + exhaust_mult * exhaust_score.values.clip(0, 2.0)
        bet_size = np.minimum(base_size * boost, max_bet)
        bet_size = np.where(signal.values != 0, np.round(bet_size, 2), 0.0)
        size = pd.Series(bet_size, index=candles.index)

        return pd.DataFrame(
            {
                "signal": signal,
                "size": size,
                # Diagnostics
                "streak": streak,
                "streak_dir": streak_dir,
                "tfi_exhaust": tfi_exhaust,
                "mp_exhaust": mp_exhaust,
                "exhaust_score": exhaust_score,
                "hawkes_intensity": hawkes,
                "regime_ok": regime_ok.astype(int),
            },
            index=candles.index,
        )
