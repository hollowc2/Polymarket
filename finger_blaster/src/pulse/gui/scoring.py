"""Signal scoring: converts IndicatorSnapshot to 0-100 score with bullish/bearish label."""

from typing import Tuple
from src.pulse.config import IndicatorSnapshot


def compute_signal_score(snapshot: IndicatorSnapshot) -> Tuple[int, str, str]:
    """Compute (score, label, description) from indicators. Score 0-100: 0-40=bearish, 41-60=neutral, 61-100=bullish."""
    components = []

    # 1. RSI Component (weight: 25%)
    if snapshot.rsi is not None:
        rsi_score = _compute_rsi_score(snapshot.rsi)
        components.append(("RSI", rsi_score, 0.25))

    # 2. MACD Component (weight: 25%)
    if snapshot.macd_histogram is not None:
        macd_score = _compute_macd_score(snapshot.macd_histogram)
        components.append(("MACD", macd_score, 0.25))

    # 3. Trend Direction from ADX/DI (weight: 25%)
    if snapshot.trend_direction:
        trend_score = _compute_trend_score(snapshot)
        components.append(("Trend", trend_score, 0.25))

    # 4. VWAP Deviation (weight: 25%)
    if snapshot.vwap_deviation is not None:
        vwap_score = _compute_vwap_score(snapshot.vwap_deviation)
        components.append(("VWAP", vwap_score, 0.25))

    # Calculate weighted average
    if not components:
        return (50, "No Data", "Insufficient indicator data")

    total_weight = sum(c[2] for c in components)
    weighted_sum = sum(c[1] * c[2] for c in components)
    score = int(weighted_sum / total_weight)

    # Clamp to 0-100 range
    score = max(0, min(100, score))

    # Generate label and description based on score
    label, description = _get_label_and_description(score)

    return (score, label, description)


def _compute_rsi_score(rsi: float) -> float:
    """RSI score: >70=bearish (overbought), <30=bullish (oversold), 30-70=neutral."""
    if rsi > 70:
        # Overbought: higher RSI = more bearish
        # At RSI=70, score=50; at RSI=100, score=0
        return max(0, 50 - (rsi - 70) * (50 / 30))
    elif rsi < 30:
        # Oversold: lower RSI = more bullish
        # At RSI=30, score=50; at RSI=0, score=100
        return min(100, 50 + (30 - rsi) * (50 / 30))
    else:
        # Neutral zone: RSI maps linearly to 40-60
        # RSI=50 -> score=50
        return 40 + (rsi - 30) * (20 / 40)


def _compute_macd_score(macd_histogram: float) -> float:
    """MACD score: positive=bullish (>50), negative=bearish (<50)."""
    if macd_histogram > 0:
        # Bullish: stronger positive = higher score
        # Map magnitude to 50-100 range
        return min(100, 50 + abs(macd_histogram) * 100)
    else:
        # Bearish: stronger negative = lower score
        # Map magnitude to 0-50 range
        return max(0, 50 - abs(macd_histogram) * 100)


def _compute_trend_score(snapshot: IndicatorSnapshot) -> float:
    """Trend score: UP=bullish, DOWN=bearish, SIDEWAYS=neutral, amplified by ADX."""
    base_scores = {
        "UP": 75,
        "DOWN": 25,
        "SIDEWAYS": 50,
    }

    trend_score = base_scores.get(snapshot.trend_direction, 50)

    # Amplify by ADX strength (0-100)
    if snapshot.adx is not None and snapshot.adx > 0:
        # ADX > 25 = strong trend, amplify deviation from 50
        adx_factor = min(snapshot.adx / 25, 2.0)  # Cap at 2x amplification
        trend_score = 50 + (trend_score - 50) * adx_factor

    return max(0, min(100, trend_score))


def _compute_vwap_score(vwap_deviation: float) -> float:
    """VWAP score: above=bullish, below=bearish, at=neutral."""
    if vwap_deviation > 0:
        # Above VWAP: bullish
        # deviation=1 -> score=60; deviation=5 -> score=100
        return min(100, 50 + vwap_deviation * 10)
    else:
        # Below VWAP: bearish
        # deviation=-1 -> score=40; deviation=-5 -> score=0
        return max(0, 50 + vwap_deviation * 10)


def _get_label_and_description(score: int) -> Tuple[str, str]:
    """Get label and description based on score."""
    if score >= 80:
        return ("Strong Buy", "Strong bullish")
    elif score >= 65:
        return ("Bullish", "Bullish momentum")
    elif score >= 55:
        return ("Lean Bull", "Slight bullish")
    elif score >= 45:
        return ("Neutral", "Mixed/choppy")
    elif score >= 35:
        return ("Lean Bear", "Slight bearish")
    elif score >= 20:
        return ("Bearish", "Bearish momentum")
    else:
        return ("Strong Sell", "Strong bearish")
