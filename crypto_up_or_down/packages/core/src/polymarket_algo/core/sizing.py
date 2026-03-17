"""Bet sizing utilities — Kelly criterion and historical reversal rates."""

from dataclasses import dataclass

# Measured reversal rates per timeframe — 2-year Binance BTCUSDT backtest (210k candles).
# 95% Wilson CI for reference:
#   5m : ±0.3% at trigger=2, ±0.6% at trigger=4, ±3.0% at trigger=8
#   15m: ±0.5% at trigger=2, ±1.2% at trigger=4, ±6.7% at trigger=8
#   1h : ±1.1% at trigger=2, ±2.3% at trigger=4, ±10.4% at trigger=8 (thin at high triggers)
# Note: 1h trigger≥7 drops below 50% — Kelly will correctly size those to 0.
REVERSAL_RATES: dict[str, dict[int, float]] = {
    "5m": {
        2: 0.518,
        3: 0.527,
        4: 0.537,
        5: 0.533,
        6: 0.540,
        7: 0.550,
        8: 0.562,
    },
    "15m": {
        2: 0.537,
        3: 0.556,
        4: 0.568,
        5: 0.579,
        6: 0.603,
        7: 0.598,
        8: 0.615,
    },
    "1h": {
        2: 0.541,
        3: 0.548,
        4: 0.557,
        5: 0.546,
        6: 0.533,
        7: 0.479,
        8: 0.482,
    },
}

# Best trigger per timeframe — highest train Sharpe with a reliable sample size.
# 5m : trigger=4 (Sharpe 3.19, 18k train trades)
# 15m: trigger=6 (Sharpe 6.01,  919 train trades)
# 1h : trigger=4 (Sharpe 2.76, 1.3k train trades, only trigger that survives test)
DEFAULT_TRIGGERS: dict[str, int] = {"5m": 4, "15m": 6, "1h": 4}


@dataclass(frozen=True)
class RateEstimate:
    """95% Wilson score CI for a measured reversal rate."""

    rate: float  # point estimate (measured win rate)
    ci_lo: float  # 95% Wilson lower bound
    ci_hi: float  # 95% Wilson upper bound
    n_trades: int  # sample size

    @property
    def conservative(self) -> float:
        """Lower CI bound — smaller bet when sample is thin (wide CI)."""
        return self.ci_lo

    @property
    def ci_width(self) -> float:
        return self.ci_hi - self.ci_lo


# Per-asset 5m reversal rates — 730-day Binance backtest (2024-02-29 → 2026-02-28).
# Wilson score CI z=1.96, ~210k candles per asset.
ASSET_REVERSAL_RATES: dict[str, dict[str, dict[int, RateEstimate]]] = {
    "BTC": {
        "5m": {
            2: RateEstimate(0.519, 0.516, 0.522, 104241),
            3: RateEstimate(0.529, 0.525, 0.533, 50163),
            4: RateEstimate(0.540, 0.533, 0.546, 23629),
            5: RateEstimate(0.538, 0.528, 0.547, 10881),
            6: RateEstimate(0.543, 0.529, 0.556, 5032),
            7: RateEstimate(0.553, 0.533, 0.573, 2302),
            8: RateEstimate(0.562, 0.531, 0.592, 1029),
        },
    },
    "ETH": {
        "5m": {
            2: RateEstimate(0.526, 0.523, 0.529, 103547),
            3: RateEstimate(0.536, 0.532, 0.541, 49053),
            4: RateEstimate(0.550, 0.543, 0.556, 22746),
            5: RateEstimate(0.557, 0.547, 0.567, 10240),
            6: RateEstimate(0.554, 0.540, 0.569, 4536),
            7: RateEstimate(0.542, 0.521, 0.564, 2021),
            8: RateEstimate(0.544, 0.512, 0.576, 925),
        },
    },
    "SOL": {
        "5m": {
            2: RateEstimate(0.517, 0.514, 0.520, 104649),
            3: RateEstimate(0.525, 0.520, 0.529, 50551),
            4: RateEstimate(0.535, 0.529, 0.542, 24033),
            5: RateEstimate(0.544, 0.534, 0.553, 11166),
            6: RateEstimate(0.547, 0.533, 0.561, 5096),
            7: RateEstimate(0.544, 0.524, 0.564, 2309),
            8: RateEstimate(0.541, 0.511, 0.571, 1053),
        },
    },
    "XRP": {
        "5m": {
            2: RateEstimate(0.516, 0.513, 0.519, 104377),
            3: RateEstimate(0.528, 0.523, 0.532, 50526),
            4: RateEstimate(0.532, 0.526, 0.538, 23868),
            5: RateEstimate(0.538, 0.529, 0.548, 11169),
            6: RateEstimate(0.545, 0.531, 0.559, 5156),
            7: RateEstimate(0.547, 0.527, 0.567, 2346),
            8: RateEstimate(0.534, 0.504, 0.564, 1062),
        },
    },
}


def get_reversal_rate(timeframe: str, trigger: int, asset: str = "") -> float:
    """Return the measured win rate for a timeframe + trigger combination.

    If `asset` is provided and present in ASSET_REVERSAL_RATES, returns the
    asset-specific point estimate. Otherwise falls back to the legacy REVERSAL_RATES
    table (BTC-derived). Clamps trigger to the highest key in the table.

    Backward compatible: calling with two args (no asset) is identical to before.
    """
    if asset and asset in ASSET_REVERSAL_RATES:
        tf_data = ASSET_REVERSAL_RATES[asset].get(timeframe)
        if tf_data:
            return tf_data[min(trigger, max(tf_data))].rate
    rates = REVERSAL_RATES.get(timeframe, REVERSAL_RATES["5m"])
    return rates[min(trigger, max(rates))]


def get_rate_estimate(timeframe: str, trigger: int, asset: str = "") -> "RateEstimate | None":
    """Return the full RateEstimate (point + CI + n) for an asset/timeframe/trigger.

    Returns None if the asset is unknown or has no data for the given timeframe.
    """
    if asset and asset in ASSET_REVERSAL_RATES:
        tf_data = ASSET_REVERSAL_RATES[asset].get(timeframe)
        if tf_data:
            return tf_data[min(trigger, max(tf_data))]
    return None


@dataclass
class BetDecision:
    """Result of evaluating a strategy signal for execution."""

    should_bet: bool
    direction: str  # "up" or "down" — the side to bet ON
    size: float  # USD amount
    confidence: float  # estimated win probability (0-1)
    reason: str


def kelly_size(
    confidence: float,
    odds: float,
    bankroll: float,
    fraction: float = 0.25,
) -> float:
    """Calculate bet size using fractional Kelly criterion.

    Args:
        confidence: Estimated win probability (0-1)
        odds: Decimal odds (e.g., 2.0 for even money at 50 cents)
        bankroll: Current bankroll in USD
        fraction: Kelly fraction (0.25 = quarter Kelly, conservative)

    Returns:
        Recommended bet size in USD (minimum $1 if positive edge)
    """
    if confidence <= 0 or odds <= 1:
        return 0

    # Kelly formula: f* = (bp - q) / b
    # where b = odds - 1, p = win prob, q = 1 - p
    b = odds - 1
    p = confidence
    q = 1 - p

    kelly = (b * p - q) / b
    if kelly <= 0:
        return 0

    return max(1, round(bankroll * kelly * fraction, 2))
