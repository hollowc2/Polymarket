"""ApexML — Walk-forward Logistic Regression on multi-TF momentum + microstructure.

Strategy Protocol implementation that learns optimal weights across 12 features
using logistic regression, walk-forward validated (train pre-2024, test 2024+).

Key research findings:
    Feature      corr_fwd   notes
    mom_15m      +0.2179    Primary signal — 15m momentum PREDICTS next 5m bar
    mom_60m      +0.1465    Secondary momentum signal
    cvd_15m_z    +0.1298    Buy/sell pressure at 15m confirms direction
    cvd_60m_z    +0.0931    Weaker but consistent
    5m TFI/OBI   ~0.025     Baseline microstructure (reversal biased)
    trade_count  -0.0016    Noise — L2 regularization suppresses automatically

Design
------
1. fit(train_candles)    — trains LogisticRegression(C=1.0) on scaled features
2. save(path)            — serializes coefs + scaler to JSON
3. load(path)            — restores from JSON for live use
4. evaluate(candles)     — Strategy Protocol: returns signal/size DataFrame

Inference uses only numpy (no sklearn at inference time).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .apex_ml_features import compute_all_features


class ApexMLStrategy:
    """Logistic regression strategy on multi-TF momentum + microstructure.

    Workflow:
        1. fit(train_candles)       — train on historical data
        2. save(path)               — serialize weights to JSON
        3. load(path)               — restore from JSON for live use
        4. evaluate(candles, ...)   — Strategy Protocol inference

    The model stores only numpy arrays (coefs + scaler params), so inference
    is a single matrix multiply — no scikit-learn import needed at runtime.
    """

    name = "apex_ml"
    description = (
        "Walk-forward logistic regression: multi-TF momentum (15m/60m) + "
        "microstructure features. mom_15m dominates the signal."
    )
    timeframe = "5m"

    def __init__(self) -> None:
        self._coefs: np.ndarray = np.zeros(12)
        self._intercept: float = 0.0
        self._scaler_mean: np.ndarray = np.zeros(12)
        self._scaler_std: np.ndarray = np.ones(12)
        self._feature_names: list[str] = []
        self._fitted: bool = False

    @property
    def default_params(self) -> dict:
        return {
            "edge_threshold": 0.02,
            "kelly_scale": 0.25,
            "max_bet": 15.0,
        }

    @property
    def param_grid(self) -> dict[str, list]:
        return {
            "edge_threshold": [0.01, 0.02, 0.03, 0.04],
            "kelly_scale": [0.20, 0.30],
            "max_bet": [10.0, 15.0, 20.0],
        }

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, candles: pd.DataFrame, C: float = 1.0, max_iter: int = 500) -> None:
        """Train LogisticRegression on candle history.

        StandardScaler is fit on the training data only (no leakage). The
        scaler params are saved alongside the model weights for inference.

        Args:
            candles:  Full training OHLCV DataFrame (enriched with CVD preferred).
            C:        Inverse regularization strength (L2). Smaller = more regularized.
            max_iter: Max solver iterations.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        feat = compute_all_features(candles)

        # Target: +1 if next bar up, -1 if down (shift -1, drop last bar)
        direction = (candles["close"].diff() > 0).astype(int) * 2 - 1  # +1/-1
        target = direction.shift(-1).dropna()  # next bar direction

        # Align features and target (drop last row — it has no label)
        feat = feat.loc[target.index]
        self._feature_names = list(feat.columns)

        X = feat.values.astype(float)
        y = target.values.astype(int)

        # Fit scaler on training data only
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scaler_mean = scaler.mean_.copy()  # type: ignore[union-attr]
        self._scaler_std = scaler.scale_.copy()  # type: ignore[union-attr]

        # Fit logistic regression with L2 regularization
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        clf.fit(X_scaled, y)

        self._coefs = clf.coef_[0].copy()
        self._intercept = float(clf.intercept_[0])
        self._fitted = True

    # ------------------------------------------------------------------ #
    # Inference — numpy only, no sklearn                                  #
    # ------------------------------------------------------------------ #

    def _predict_proba_up(self, X_raw: np.ndarray) -> np.ndarray:
        """Predict P(UP) using stored weights. No sklearn import."""
        # Guard against zero std
        std = np.where(self._scaler_std > 0, self._scaler_std, 1.0)
        X_scaled = (X_raw - self._scaler_mean) / std
        logit = X_scaled @ self._coefs + self._intercept
        return 1.0 / (1.0 + np.exp(-logit.clip(-20, 20)))

    def evaluate(self, candles: pd.DataFrame, **params) -> pd.DataFrame:
        """Strategy Protocol evaluate(). Returns DataFrame[signal, size].

        Args:
            candles:  OHLCV DataFrame (5m, DatetimeTZDtype UTC index).
            **params: edge_threshold, kelly_scale, max_bet.

        Returns:
            DataFrame with columns:
                signal   int    +1 (UP), -1 (DOWN), 0 (no trade)
                size     float  USD bet size (0 when signal == 0)
                prob_up  float  P(UP) diagnostic
                edge     float  P(UP) - 0.5 diagnostic
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        p = {**self.default_params, **params}
        edge_threshold = float(p["edge_threshold"])
        kelly_scale = float(p["kelly_scale"])
        max_bet = float(p["max_bet"])

        feat = compute_all_features(candles)
        X = feat.values.astype(float)

        prob_up = self._predict_proba_up(X)
        edge = prob_up - 0.5

        # Signal: threshold on edge
        signal_vals = np.where(
            edge > edge_threshold,
            1,
            np.where(edge < -edge_threshold, -1, 0),
        )

        # Kelly-fractional sizing
        # Kelly fraction for binary bet: f = edge / p(1-p)
        # Scaled by kelly_scale and capped at max_bet
        p_clip = prob_up.clip(0.01, 0.99)
        kelly_size = np.minimum(
            kelly_scale * max_bet * np.abs(edge) / (p_clip * (1 - p_clip)),
            max_bet,
        )
        size_vals = np.where(signal_vals != 0, np.round(kelly_size, 2), 0.0)

        return pd.DataFrame(
            {
                "signal": signal_vals.astype(int),
                "size": size_vals,
                # Diagnostics
                "prob_up": prob_up,
                "edge": edge,
            },
            index=candles.index,
        )

    # ------------------------------------------------------------------ #
    # Serialization — JSON                                               #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialize model weights and scaler params to JSON."""
        if not self._fitted:
            raise RuntimeError("Model not fitted — nothing to save.")

        data: dict[str, Any] = {
            "name": self.name,
            "feature_names": self._feature_names,
            "coefs": self._coefs.tolist(),
            "intercept": self._intercept,
            "scaler_mean": self._scaler_mean.tolist(),
            "scaler_std": self._scaler_std.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ApexMLStrategy:
        """Load model from JSON file."""
        with open(path) as f:
            data: dict[str, Any] = json.load(f)

        model = cls()
        model._feature_names = data["feature_names"]
        model._coefs = np.array(data["coefs"], dtype=float)
        model._intercept = float(data["intercept"])
        model._scaler_mean = np.array(data["scaler_mean"], dtype=float)
        model._scaler_std = np.array(data["scaler_std"], dtype=float)
        model._fitted = True
        return model

    def feature_importance(self) -> dict[str, float]:
        """Return feature name → coefficient dict, sorted by magnitude descending."""
        if not self._fitted:
            return {}
        return dict(
            sorted(
                zip(self._feature_names, self._coefs.tolist(), strict=False),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        )
