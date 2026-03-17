"""APEX Strategy — Probabilistic Microstructure Trading.

Combines six order-flow and microstructure signals into a Bayesian log-odds
estimate.  The resulting edge drives Kelly-fractional position sizing with an
inventory penalty that shrinks exposure as directional risk accumulates.

Signal stack
------------
TFI              Trade Flow Imbalance     (from Binance CVD buy/sell volume)
OBI_proxy        Order Book Imbalance     (rolling TFI z-score approximation)
microprice_drift Microprice − midprice    (ATR-normalised buy/sell pressure)
hawkes_intensity Hawkes self-excitation   (clustered arrival intensity ∈ (0,1))
cascade_score    Liquidation cascade      (signed severity from liq_* columns)
funding_pressure Funding × momentum × OI  (positioning directional pressure)

Edge model
----------
log_odds = w_obi   × OBI_proxy
         + w_tfi   × TFI
         + w_mp    × microprice_drift
         + w_hwk   × (hawkes_intensity − 0.5)   ← centred around 0
         + w_cas   × cascade_score
         + w_fund  × funding_pressure

p_model = sigmoid(log_odds)
edge    = p_model − p_market          (default p_market = 0.50)

Signal generation
-----------------
signal = +1   if edge > +edge_threshold
signal = -1   if edge < -edge_threshold
signal =  0   otherwise

Cascade mode  (when |cascade_score| > cascade_threshold)
------------
Edge threshold is halved and position sizing is boosted up to cascade_size_mult×
to exploit the directional momentum from a liquidation cascade event.

Position sizing (Kelly-fractional)
----------------------------------
f*   = |edge| / variance        full Kelly fraction
size = f* × kelly_scale × max_bet × inventory_penalty

inventory_penalty = 1 − (|position| / max_position)^2

Regime and adverse-selection gates
-----------------------------------
When regime_ok == False or toxicity > toxicity_threshold the signal is zeroed.

Requires
--------
enrich_candles() from polymarket_algo.data — adds buy_vol, sell_vol, delta,
cvd, liq_long_usd, liq_short_usd, funding_rate, funding_zscore columns.

Falls back gracefully on plain OHLCV (lower signal quality, no CVD/liq/funding
features — strategy still runs but with reduced feature coverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .apex_features import (
    compute_cascade_score,
    compute_funding_pressure,
    compute_hawkes_intensity,
    compute_microprice_drift,
    compute_obi_proxy,
    compute_tfi,
)
from .apex_filters import compute_adverse_selection, detect_regime

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable element-wise sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))


def _kelly_size(
    edge: float,
    variance: float,
    kelly_scale: float,
    max_bet: float,
    inventory_factor: float,
) -> float:
    """Kelly-fractional position size with inventory penalty.

    f* = |edge| / variance  (full Kelly)
    size = min(f* × kelly_scale × max_bet, max_bet) × inventory_factor
    """
    if variance <= 0.0 or abs(edge) < 1e-6:
        return 0.0
    f_full = abs(edge) / variance
    size = min(f_full * kelly_scale * max_bet, max_bet)
    return float(size * inventory_factor)


# ---------------------------------------------------------------------------
# Main Strategy
# ---------------------------------------------------------------------------


class ApexStrategy:
    """APEX probabilistic microstructure strategy (Strategy Protocol).

    Implements the Strategy Protocol:
        name, description, timeframe
        default_params, param_grid
        evaluate(candles, **params) → DataFrame[signal, size]

    Parameters (all overridable via evaluate(**params) or param_grid)
    ----------
    w_obi            float  Weight for OBI proxy signal        (default 0.80)
    w_tfi            float  Weight for Trade Flow Imbalance    (default 0.80)
    w_mp             float  Weight for microprice drift        (default 0.60)
    w_hwk            float  Weight for Hawkes intensity        (default 0.40)
    w_cas            float  Weight for cascade score           (default 0.60)
    w_fund           float  Weight for funding pressure        (default 0.50)
    edge_threshold   float  Minimum |edge| to generate signal  (default 0.04)
    toxicity_max     float  Max toxicity before vetoing signal  (default 0.65)
    kelly_scale      float  Fraction of full Kelly to use       (default 0.25)
    max_bet          float  Maximum USD bet size                (default 15.0)
    current_position float  Current net exposure (USD, signed)  (default 0.0)
    max_position     float  Max USD exposure for inventory ctrl (default 50.0)
    cascade_threshold float  |cascade_score| above which cascade mode activates
                            (default 1.5)
    cascade_size_mult float  Size multiplier during cascade mode (default 1.5)
    hawkes_mu        float  Hawkes background intensity         (default 0.10)
    hawkes_alpha     float  Hawkes self-excitation coeff        (default 0.30)
    hawkes_beta      float  Hawkes memory decay per candle      (default 1.00)
    obi_window       int    OBI proxy rolling window (candles)  (default 12)
    p_market         float  Market implied probability (0.5 for symmetric Poly
                            markets; can be updated from live order book prices)
                            (default 0.50)
    """

    name = "apex"
    description = (
        "Probabilistic microstructure bot: Bayesian edge from OBI, TFI, "
        "microprice drift, Hawkes intensity, liquidation cascades, and funding pressure"
    )
    timeframe = "5m"

    @property
    def default_params(self) -> dict:
        return {
            # --- Signal weights ---
            "w_obi": 0.80,
            "w_tfi": 0.80,
            "w_mp": 0.60,
            "w_hwk": 0.40,
            "w_cas": 0.60,
            "w_fund": 0.50,
            # --- Entry threshold ---
            "edge_threshold": 0.04,
            # --- Adverse selection ---
            "toxicity_max": 0.65,
            # --- Sizing ---
            "kelly_scale": 0.25,
            "max_bet": 15.0,
            # --- Inventory ---
            "current_position": 0.0,
            "max_position": 50.0,
            # --- Cascade mode ---
            "cascade_threshold": 1.5,
            "cascade_size_mult": 1.5,
            # --- Hawkes tuning ---
            "hawkes_mu": 0.10,
            "hawkes_alpha": 0.30,
            "hawkes_beta": 1.00,
            # --- OBI proxy window ---
            "obi_window": 12,
            # --- Market implied probability ---
            "p_market": 0.50,
        }

    @property
    def param_grid(self) -> dict[str, list]:
        """Grid for parameter sweep / walk-forward calibration."""
        return {
            "w_obi": [0.4, 0.8, 1.2],
            "w_tfi": [0.4, 0.8, 1.2],
            "w_mp": [0.3, 0.6, 0.9],
            "w_hwk": [0.2, 0.4, 0.6],
            "w_cas": [0.3, 0.6, 0.9],
            "w_fund": [0.2, 0.5, 0.8],
            "edge_threshold": [0.02, 0.04, 0.06],
            "toxicity_max": [0.55, 0.65, 0.75],
            "kelly_scale": [0.15, 0.25, 0.35],
            "max_bet": [10.0, 15.0, 20.0],
        }

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Evaluate APEX strategy on a candle DataFrame.

        Args:
            candles: OHLCV DataFrame indexed by open_time (UTC).
                     Ideally enriched via enrich_candles() for full signal
                     coverage; degrades gracefully on plain OHLCV.
            **params: Any parameter from default_params / param_grid.

        Returns:
            DataFrame with columns:
                signal  int    1 (UP), -1 (DOWN), 0 (no trade)
                size    float  USD bet size (0 when signal == 0)

            Additional diagnostic columns are included for inspection:
                edge, p_model, log_odds, toxicity, regime_ok,
                cascade_mode, tfi, obi_proxy, microprice_drift,
                hawkes_intensity, cascade_score, funding_pressure
        """
        p = {**self.default_params, **params}

        w_obi = float(p["w_obi"])
        w_tfi = float(p["w_tfi"])
        w_mp = float(p["w_mp"])
        w_hwk = float(p["w_hwk"])
        w_cas = float(p["w_cas"])
        w_fund = float(p["w_fund"])

        edge_threshold = float(p["edge_threshold"])
        toxicity_max = float(p["toxicity_max"])
        kelly_scale = float(p["kelly_scale"])
        max_bet = float(p["max_bet"])
        current_position = float(p["current_position"])
        max_position = float(p["max_position"])
        cascade_threshold = float(p["cascade_threshold"])
        cascade_size_mult = float(p["cascade_size_mult"])
        hawkes_mu = float(p["hawkes_mu"])
        hawkes_alpha = float(p["hawkes_alpha"])
        hawkes_beta = float(p["hawkes_beta"])
        obi_window = int(p["obi_window"])
        p_market = float(p["p_market"])

        # ── Feature computation ──────────────────────────────────────
        tfi = compute_tfi(candles)
        obi = compute_obi_proxy(candles, window=obi_window)
        mp_drift = compute_microprice_drift(candles)
        hawkes = compute_hawkes_intensity(candles, mu=hawkes_mu, alpha=hawkes_alpha, beta=hawkes_beta)
        cascade = compute_cascade_score(candles)
        funding = compute_funding_pressure(candles)

        # ── Adverse selection and regime ─────────────────────────────
        toxicity = compute_adverse_selection(candles)
        regime_ok = detect_regime(candles)

        # ── Bayesian log-odds ────────────────────────────────────────
        # Hawkes intensity is in (0,1); centre it around 0 so it acts as a
        # signed excitation boost rather than a directional signal.
        hawkes_centred = hawkes - 0.5

        log_odds = (
            w_obi * obi + w_tfi * tfi + w_mp * mp_drift + w_hwk * hawkes_centred + w_cas * cascade + w_fund * funding
        )

        p_model = pd.Series(_sigmoid(log_odds.values), index=candles.index, name="p_model")
        edge_series = (p_model - p_market).rename("edge")

        # ── Cascade mode flag ────────────────────────────────────────
        cascade_mode = cascade.abs() > cascade_threshold

        # ── Inventory penalty (scalar — position fixed for the whole batch) ──
        inv_ratio = abs(current_position) / max(max_position, 1e-9)
        inventory_factor = float(np.clip(1.0 - inv_ratio**2, 0.0, 1.0))

        # ── Vectorised signal generation ──────────────────────────────
        casc_vals = cascade_mode.values
        edge_vals = edge_series.values

        # Cascade-aware threshold: halved on cascade bars
        eff_thresh = np.where(casc_vals, edge_threshold * 0.5, edge_threshold)

        # Gate mask: must pass regime + toxicity filters
        gate_ok = regime_ok.values & (toxicity.values <= toxicity_max)

        long_mask = gate_ok & (edge_vals > eff_thresh)
        short_mask = gate_ok & (edge_vals < -eff_thresh)

        # Inventory block at saturation
        if inv_ratio >= 1.0:
            if current_position > 0:
                long_mask = np.zeros_like(long_mask)
            elif current_position < 0:
                short_mask = np.zeros_like(short_mask)

        raw_sig = np.where(long_mask, 1, np.where(short_mask, -1, 0))

        # ── Vectorised Kelly sizing ───────────────────────────────────
        p_vals = p_model.values
        variance = np.maximum(p_vals * (1.0 - p_vals), 1e-6)
        abs_edge = np.abs(edge_vals)
        f_star = np.where(abs_edge > 1e-6, abs_edge / variance, 0.0)
        base_size = np.minimum(f_star * kelly_scale * max_bet, max_bet) * inventory_factor

        # Cascade boost, capped at cascade_size_mult × max_bet
        casc_cap = max_bet * cascade_size_mult
        bet_size = np.where(
            casc_vals & (raw_sig != 0),
            np.minimum(base_size * cascade_size_mult, casc_cap),
            base_size,
        )

        active = (raw_sig != 0) & (bet_size >= 1.0)
        signal = pd.Series(np.where(active, raw_sig, 0).astype(int), index=candles.index)
        size = pd.Series(np.where(active, np.round(bet_size, 2), 0.0), index=candles.index)

        # ── Assemble output ──────────────────────────────────────────
        out = pd.DataFrame(
            {
                "signal": signal,
                "size": size,
                # Diagnostic columns
                "edge": edge_series,
                "p_model": p_model,
                "log_odds": log_odds,
                "toxicity": toxicity,
                "regime_ok": regime_ok.astype(int),
                "cascade_mode": cascade_mode.astype(int),
                "tfi": tfi,
                "obi_proxy": obi,
                "microprice_drift": mp_drift,
                "hawkes_intensity": hawkes,
                "cascade_score": cascade,
                "funding_pressure": funding,
            },
            index=candles.index,
        )
        return out
