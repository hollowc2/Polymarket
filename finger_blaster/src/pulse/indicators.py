"""Technical indicator calculations for Pulse."""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional

from src.pulse.config import (
    Alert,
    Candle,
    IndicatorSnapshot,
    PulseConfig,
    Timeframe,
)

logger = logging.getLogger("Pulse.Indicators")


class VWAPCalculator:
    def __init__(self, reset_hour_utc: int = 0):
        self.reset_hour_utc = reset_hour_utc

        self._cumulative_tp_volume: float = 0.0  # Sum of (typical_price * volume)
        self._cumulative_volume: float = 0.0
        self._last_reset_date: Optional[str] = None
        self._current_vwap: Optional[float] = None

    def _should_reset(self, timestamp: int) -> bool:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        current_date = dt.strftime("%Y-%m-%d")

        if self._last_reset_date is None:
            self._last_reset_date = current_date
            return True

        if current_date != self._last_reset_date and dt.hour >= self.reset_hour_utc:
            self._last_reset_date = current_date
            return True

        return False

    def update(self, candle: Candle) -> float:
        # Check for reset
        if self._should_reset(candle.timestamp):
            self.reset()

        # Calculate typical price and update cumulative values
        typical_price = candle.typical_price
        self._cumulative_tp_volume += typical_price * candle.volume
        self._cumulative_volume += candle.volume

        # Calculate VWAP
        if self._cumulative_volume > 0:
            self._current_vwap = self._cumulative_tp_volume / self._cumulative_volume
        else:
            self._current_vwap = candle.close

        return self._current_vwap

    def reset(self):
        self._cumulative_tp_volume = 0.0
        self._cumulative_volume = 0.0
        self._current_vwap = None

    @property
    def current_vwap(self) -> Optional[float]:
        return self._current_vwap

    def get_deviation(self, current_price: float) -> Optional[float]:
        if self._current_vwap is None or self._current_vwap == 0:
            return None
        return ((current_price - self._current_vwap) / self._current_vwap) * 100


class ADXCalculator:
    """ADX: >25=strong trend, <20=weak trend."""

    def __init__(self, period: int = 14):
        self.period = period

        # Price history for calculations
        self._highs: Deque[float] = deque(maxlen=period + 1)
        self._lows: Deque[float] = deque(maxlen=period + 1)
        self._closes: Deque[float] = deque(maxlen=period + 1)

        # Smoothed values
        self._smoothed_plus_dm: Optional[float] = None
        self._smoothed_minus_dm: Optional[float] = None
        self._smoothed_tr: Optional[float] = None
        self._smoothed_dx: Optional[float] = None

        # Current values
        self._current_adx: Optional[float] = None
        self._current_plus_di: Optional[float] = None
        self._current_minus_di: Optional[float] = None

    def update(self, candle: Candle) -> Optional[float]:
        self._highs.append(candle.high)
        self._lows.append(candle.low)
        self._closes.append(candle.close)

        if len(self._highs) < 2:
            return None

        # Calculate True Range
        prev_close = self._closes[-2]
        tr = max(
            candle.high - candle.low,
            abs(candle.high - prev_close),
            abs(candle.low - prev_close)
        )

        # Calculate Directional Movement
        prev_high = self._highs[-2]
        prev_low = self._lows[-2]

        up_move = candle.high - prev_high
        down_move = prev_low - candle.low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0

        # Smoothing
        if self._smoothed_tr is None:
            # First value - need period candles
            if len(self._highs) < self.period + 1:
                return None

            # Initialize with simple sum
            self._smoothed_tr = tr
            self._smoothed_plus_dm = plus_dm
            self._smoothed_minus_dm = minus_dm
        else:
            # Wilder's smoothing
            self._smoothed_tr = self._smoothed_tr - (self._smoothed_tr / self.period) + tr
            self._smoothed_plus_dm = self._smoothed_plus_dm - (self._smoothed_plus_dm / self.period) + plus_dm
            self._smoothed_minus_dm = self._smoothed_minus_dm - (self._smoothed_minus_dm / self.period) + minus_dm

        # Calculate +DI and -DI
        if self._smoothed_tr > 0:
            self._current_plus_di = 100 * (self._smoothed_plus_dm / self._smoothed_tr)
            self._current_minus_di = 100 * (self._smoothed_minus_dm / self._smoothed_tr)
        else:
            self._current_plus_di = 0.0
            self._current_minus_di = 0.0

        # Calculate DX
        di_sum = self._current_plus_di + self._current_minus_di
        if di_sum > 0:
            dx = 100 * abs(self._current_plus_di - self._current_minus_di) / di_sum
        else:
            dx = 0.0

        # Smooth ADX
        if self._smoothed_dx is None:
            self._smoothed_dx = dx
            self._current_adx = dx
        else:
            self._smoothed_dx = ((self._smoothed_dx * (self.period - 1)) + dx) / self.period
            self._current_adx = self._smoothed_dx

        return self._current_adx

    @property
    def current_adx(self) -> Optional[float]:
        return self._current_adx

    @property
    def plus_di(self) -> Optional[float]:
        return self._current_plus_di

    @property
    def minus_di(self) -> Optional[float]:
        return self._current_minus_di

    @property
    def trend_direction(self) -> str:
        if self._current_plus_di is None or self._current_minus_di is None:
            return "SIDEWAYS"

        if self._current_adx is not None and self._current_adx < 20:
            return "SIDEWAYS"

        if self._current_plus_di > self._current_minus_di:
            return "UP"
        elif self._current_minus_di > self._current_plus_di:
            return "DOWN"
        else:
            return "SIDEWAYS"

    def reset(self):
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._smoothed_plus_dm = None
        self._smoothed_minus_dm = None
        self._smoothed_tr = None
        self._smoothed_dx = None
        self._current_adx = None
        self._current_plus_di = None
        self._current_minus_di = None


class ATRCalculator:
    def __init__(self, period: int = 14):
        self.period = period

        self._prev_close: Optional[float] = None
        self._smoothed_atr: Optional[float] = None
        self._tr_values: Deque[float] = deque(maxlen=period)

    def update(self, candle: Candle) -> Optional[float]:
        # Calculate True Range
        if self._prev_close is None:
            tr = candle.high - candle.low
        else:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - self._prev_close),
                abs(candle.low - self._prev_close)
            )

        self._prev_close = candle.close
        self._tr_values.append(tr)

        # Need enough values for initial ATR
        if len(self._tr_values) < self.period:
            return None

        # Calculate ATR
        if self._smoothed_atr is None:
            # First ATR is simple average
            self._smoothed_atr = sum(self._tr_values) / self.period
        else:
            # Wilder's smoothing
            self._smoothed_atr = ((self._smoothed_atr * (self.period - 1)) + tr) / self.period

        return self._smoothed_atr

    @property
    def current_atr(self) -> Optional[float]:
        return self._smoothed_atr

    def get_atr_percent(self, current_price: float) -> Optional[float]:
        if self._smoothed_atr is None or current_price == 0:
            return None
        return (self._smoothed_atr / current_price) * 100

    def reset(self):
        self._prev_close = None
        self._smoothed_atr = None
        self._tr_values.clear()


class VolatilityCalculator:
    def __init__(self, lookback: int = 20, annualize: bool = True):
        self.lookback = lookback
        self.annualize = annualize

        self._closes: Deque[float] = deque(maxlen=lookback + 1)
        self._current_volatility: Optional[float] = None

    def update(self, candle: Candle) -> Optional[float]:
        self._closes.append(candle.close)

        if len(self._closes) < self.lookback + 1:
            return None

        # Calculate log returns
        log_returns = []
        closes_list = list(self._closes)
        for i in range(1, len(closes_list)):
            if closes_list[i - 1] > 0:
                log_return = math.log(closes_list[i] / closes_list[i - 1])
                log_returns.append(log_return)

        if len(log_returns) < 2:
            return None

        # Calculate standard deviation
        mean_return = sum(log_returns) / len(log_returns)
        squared_diffs = [(r - mean_return) ** 2 for r in log_returns]
        variance = sum(squared_diffs) / (len(squared_diffs) - 1)
        std_dev = math.sqrt(variance)

        # Annualize if requested
        if self.annualize:
            # Calculate periods per year based on actual candle timeframe
            seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000
            periods_per_year = seconds_per_year / candle.timeframe.seconds
            self._current_volatility = std_dev * math.sqrt(periods_per_year) * 100
        else:
            self._current_volatility = std_dev * 100

        return self._current_volatility

    @property
    def current_volatility(self) -> Optional[float]:
        return self._current_volatility

    def reset(self):
        self._closes.clear()
        self._current_volatility = None


class RSICalculator:
    """RSI: >70=overbought, <30=oversold."""

    def __init__(self, period: int = 14):
        self.period = period

        self._closes: Deque[float] = deque(maxlen=period + 1)
        self._current_rsi: Optional[float] = None

        # Smoothed gains/losses (Wilder's smoothing)
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None

    def update(self, candle: Candle) -> Optional[float]:
        self._closes.append(candle.close)

        if len(self._closes) < 2:
            return None

        # Calculate change
        change = self._closes[-1] - self._closes[-2]
        gain = max(0, change)
        loss = max(0, -change)

        # Initialize averages on first calculation
        if self._avg_gain is None:
            if len(self._closes) < self.period + 1:
                return None

            # Calculate initial average gains/losses
            gains = []
            losses = []
            for i in range(1, len(self._closes)):
                c = self._closes[i] - self._closes[i - 1]
                if c > 0:
                    gains.append(c)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-c)

            self._avg_gain = sum(gains) / self.period
            self._avg_loss = sum(losses) / self.period
        else:
            # Wilder's smoothing
            self._avg_gain = ((self._avg_gain * (self.period - 1)) + gain) / self.period
            self._avg_loss = ((self._avg_loss * (self.period - 1)) + loss) / self.period

        # Calculate RS and RSI
        if self._avg_loss == 0:
            self._current_rsi = 100.0 if self._avg_gain > 0 else 50.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._current_rsi = 100.0 - (100.0 / (1.0 + rs))

        return self._current_rsi

    @property
    def current_rsi(self) -> Optional[float]:
        return self._current_rsi

    def reset(self):
        self._closes.clear()
        self._avg_gain = None
        self._avg_loss = None
        self._current_rsi = None

    def clone(self) -> 'RSICalculator':
        new_calc = RSICalculator(period=self.period)
        new_calc._closes = deque(self._closes, maxlen=self.period + 1)
        new_calc._avg_gain = self._avg_gain
        new_calc._avg_loss = self._avg_loss
        new_calc._current_rsi = self._current_rsi
        return new_calc


class MACDCalculator:
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

        self._closes: Deque[float] = deque(maxlen=slow + 1)

        # EMA calculations
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._macd_line: Optional[float] = None
        self._signal_line: Optional[float] = None
        self._macd_histogram: Optional[float] = None

        # Signal line EMA values
        self._macd_values: Deque[float] = deque(maxlen=signal)

    def _calculate_ema(self, data: List[float], period: int) -> float:
        if not data or len(data) < period:
            return data[-1] if data else 0.0

        # Multiplier for EMA
        multiplier = 2.0 / (period + 1)

        # Simple average for first value
        ema = sum(data[:period]) / period

        # Calculate EMA for remaining values
        for price in data[period:]:
            ema = price * multiplier + ema * (1 - multiplier)

        return ema

    def update(self, candle: Candle) -> Optional[float]:
        self._closes.append(candle.close)

        if len(self._closes) < self.slow:
            return None

        closes_list = list(self._closes)

        # Calculate EMAs using most recent data
        self._ema_fast = self._calculate_ema(closes_list, self.fast)
        self._ema_slow = self._calculate_ema(closes_list, self.slow)

        # Calculate MACD line
        self._macd_line = self._ema_fast - self._ema_slow

        # Track MACD values for signal line
        self._macd_values.append(self._macd_line)

        # Calculate signal line (EMA of MACD)
        if len(self._macd_values) >= self.signal:
            self._signal_line = self._calculate_ema(list(self._macd_values), self.signal)
            self._macd_histogram = self._macd_line - self._signal_line
        else:
            self._signal_line = None
            self._macd_histogram = None

        return self._macd_line

    @property
    def macd_line(self) -> Optional[float]:
        return self._macd_line

    @property
    def signal_line(self) -> Optional[float]:
        return self._signal_line

    @property
    def histogram(self) -> Optional[float]:
        return self._macd_histogram

    def reset(self):
        self._closes.clear()
        self._ema_fast = None
        self._ema_slow = None
        self._macd_line = None
        self._signal_line = None
        self._macd_histogram = None
        self._macd_values.clear()


class BollingerBandsCalculator:
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

        self._closes: Deque[float] = deque(maxlen=period)

        # Current values
        self._middle_band: Optional[float] = None
        self._upper_band: Optional[float] = None
        self._lower_band: Optional[float] = None

    def update(self, candle: Candle) -> Optional[float]:
        self._closes.append(candle.close)

        if len(self._closes) < self.period:
            return None

        closes_list = list(self._closes)

        # Calculate SMA (middle band)
        self._middle_band = sum(closes_list) / self.period

        # Calculate standard deviation
        variance = sum((x - self._middle_band) ** 2 for x in closes_list) / self.period
        std = math.sqrt(variance)

        # Calculate bands
        self._upper_band = self._middle_band + (self.std_dev * std)
        self._lower_band = self._middle_band - (self.std_dev * std)

        return self._middle_band

    @property
    def middle_band(self) -> Optional[float]:
        return self._middle_band

    @property
    def upper_band(self) -> Optional[float]:
        return self._upper_band

    @property
    def lower_band(self) -> Optional[float]:
        return self._lower_band

    def get_band_width(self) -> Optional[float]:
        if self._upper_band is None or self._lower_band is None or self._middle_band is None:
            return None
        if self._middle_band == 0:
            return None
        return ((self._upper_band - self._lower_band) / self._middle_band) * 100

    def get_position(self, price: float) -> Optional[float]:
        if self._upper_band is None or self._lower_band is None:
            return None
        if self._upper_band == self._lower_band:
            return None
        return (price - self._lower_band) / (self._upper_band - self._lower_band)

    def reset(self):
        self._closes.clear()
        self._middle_band = None
        self._upper_band = None
        self._lower_band = None


class IndicatorEngine:
    def __init__(
        self,
        config: PulseConfig,
        on_indicator_update: Optional[Callable[[IndicatorSnapshot], Coroutine]] = None,
        on_alert: Optional[Callable[[Alert], Coroutine]] = None,
    ):
        self.config = config
        self.on_indicator_update = on_indicator_update
        self.on_alert = on_alert

        # Per-product, per-timeframe calculators
        # Structure: product_id -> timeframe -> calculator_instance
        self._vwap: Dict[str, Dict[Timeframe, VWAPCalculator]] = {}
        self._adx: Dict[str, Dict[Timeframe, ADXCalculator]] = {}
        self._atr: Dict[str, Dict[Timeframe, ATRCalculator]] = {}
        self._volatility: Dict[str, Dict[Timeframe, VolatilityCalculator]] = {}
        self._rsi: Dict[str, Dict[Timeframe, RSICalculator]] = {}
        self._macd: Dict[str, Dict[Timeframe, MACDCalculator]] = {}
        self._bb: Dict[str, Dict[Timeframe, BollingerBandsCalculator]] = {}

        # Per-product, per-timeframe state
        self._last_snapshots: Dict[str, Dict[Timeframe, IndicatorSnapshot]] = {}
        self._last_prices: Dict[str, Dict[Timeframe, float]] = {}
        self._volume_history: Dict[str, Dict[Timeframe, Deque[float]]] = {}

        # Current candle state (for live updates like TradingView)
        # Tracks the open/high/low of the current incomplete candle
        self._current_candle_open: Dict[str, Dict[Timeframe, float]] = {}
        self._current_candle_high: Dict[str, Dict[Timeframe, float]] = {}
        self._current_candle_low: Dict[str, Dict[Timeframe, float]] = {}

        # Live calculators (separate from official candle-close calculators)
        # These are used for intra-candle updates without corrupting the real state
        self._live_rsi: Dict[str, Dict[Timeframe, RSICalculator]] = {}

    def _ensure_calculators(self, product_id: str, timeframe: Timeframe):
        # Ensure product dictionaries exist
        if product_id not in self._vwap:
            for d in [self._vwap, self._adx, self._atr, self._volatility, self._rsi,
                      self._macd, self._bb, self._last_snapshots, self._last_prices,
                      self._volume_history, self._current_candle_open,
                      self._current_candle_high, self._current_candle_low, self._live_rsi]:
                d[product_id] = {}

        # Ensure timeframe calculators exist
        if timeframe not in self._vwap[product_id]:
            cfg = self.config
            self._vwap[product_id][timeframe] = VWAPCalculator(reset_hour_utc=cfg.vwap_reset_hour_utc)
            self._adx[product_id][timeframe] = ADXCalculator(period=cfg.adx_period)
            self._atr[product_id][timeframe] = ATRCalculator(period=cfg.atr_period)
            self._volatility[product_id][timeframe] = VolatilityCalculator(lookback=cfg.volatility_lookback)
            self._rsi[product_id][timeframe] = RSICalculator(period=cfg.rsi_period)
            self._macd[product_id][timeframe] = MACDCalculator(
                fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal
            )
            self._bb[product_id][timeframe] = BollingerBandsCalculator(
                period=cfg.bb_period, std_dev=cfg.bb_std_dev
            )
            self._volume_history[product_id][timeframe] = deque(maxlen=20)
            self._live_rsi[product_id][timeframe] = RSICalculator(period=cfg.rsi_period)

    async def update(self, product_id: str, candle: Candle) -> IndicatorSnapshot:
        timeframe = candle.timeframe
        self._ensure_calculators(product_id, timeframe)

        # Update all calculators
        vwap = self._vwap[product_id][timeframe].update(candle)
        adx = self._adx[product_id][timeframe].update(candle)
        atr = self._atr[product_id][timeframe].update(candle)
        volatility = self._volatility[product_id][timeframe].update(candle)
        rsi = self._rsi[product_id][timeframe].update(candle)
        macd_line = self._macd[product_id][timeframe].update(candle)
        bb_middle = self._bb[product_id][timeframe].update(candle)

        # Track volume
        self._volume_history[product_id][timeframe].append(candle.volume)
        self._last_prices[product_id][timeframe] = candle.close

        # Reset current candle tracking (new candle just closed, start tracking next one)
        self._current_candle_open[product_id][timeframe] = candle.close  # Next candle opens at this close
        self._current_candle_high[product_id][timeframe] = candle.close
        self._current_candle_low[product_id][timeframe] = candle.close

        # Sync live RSI calculator with official one (candle just closed)
        # This keeps the live calculator in sync for accurate intra-candle updates
        self._live_rsi[product_id][timeframe] = self._rsi[product_id][timeframe].clone()

        # Calculate additional metrics
        vwap_deviation = None
        if vwap is not None:
            vwap_deviation = self._vwap[product_id][timeframe].get_deviation(candle.close)

        atr_pct = None
        if atr is not None and candle.close > 0:
            atr_pct = (atr / candle.close) * 100

        # Determine trend direction
        trend_direction = self._adx[product_id][timeframe].trend_direction

        # Determine market regime
        regime = self._determine_regime(product_id, timeframe, adx, volatility)

        # Get MACD values
        macd_signal = self._macd[product_id][timeframe].signal_line
        macd_histogram = self._macd[product_id][timeframe].histogram

        # Get Bollinger Bands values
        bb_upper = self._bb[product_id][timeframe].upper_band
        bb_lower = self._bb[product_id][timeframe].lower_band

        # Create snapshot with timeframe context
        snapshot = IndicatorSnapshot(
            product_id=product_id,
            timeframe=timeframe,
            timestamp=time.time(),
            vwap=vwap,
            vwap_deviation=vwap_deviation,
            adx=adx,
            plus_di=self._adx[product_id][timeframe].plus_di,
            minus_di=self._adx[product_id][timeframe].minus_di,
            trend_direction=trend_direction,
            atr=atr,
            atr_pct=atr_pct,
            volatility=volatility,
            regime=regime,
            rsi=rsi,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
        )

        # Check for alerts
        await self._check_alerts(product_id, timeframe, snapshot, candle)

        # Store snapshot
        self._last_snapshots[product_id][timeframe] = snapshot

        # Emit update
        if self.on_indicator_update:
            await self.on_indicator_update(snapshot)

        return snapshot

    async def update_live_price(self, product_id: str, timeframe: Timeframe, live_price: float) -> Optional[IndicatorSnapshot]:
        """TradingView-style intra-candle updates."""
        self._ensure_calculators(product_id, timeframe)

        # Get last snapshot for this timeframe (from last candle close)
        last_snapshot = self._last_snapshots.get(product_id, {}).get(timeframe)
        if not last_snapshot:
            # No baseline data yet, can't create live update
            return None

        # Get current candle tracking state
        candle_open = self._current_candle_open.get(product_id, {}).get(timeframe)
        candle_high = self._current_candle_high.get(product_id, {}).get(timeframe)
        candle_low = self._current_candle_low.get(product_id, {}).get(timeframe)

        # Initialize if not set (first update)
        if candle_open is None:
            last_close = self._last_prices.get(product_id, {}).get(timeframe)
            if last_close is None:
                return None
            candle_open = last_close
            candle_high = last_close
            candle_low = last_close
            self._current_candle_open[product_id][timeframe] = candle_open
            self._current_candle_high[product_id][timeframe] = candle_high
            self._current_candle_low[product_id][timeframe] = candle_low

        # Update high/low with live price (this tracks the actual high/low of current candle)
        if live_price > candle_high:
            candle_high = live_price
            self._current_candle_high[product_id][timeframe] = candle_high

        if live_price < candle_low:
            candle_low = live_price
            self._current_candle_low[product_id][timeframe] = candle_low

        # Create a synthetic partial candle with current live price
        # This is how TradingView does it - use actual OHLC of the current incomplete candle
        partial_candle = Candle(
            timestamp=int(time.time()),
            open=candle_open,      # Open is where this candle started
            high=candle_high,      # Actual high reached during this candle
            low=candle_low,        # Actual low reached during this candle
            close=live_price,      # Current live price is the close
            volume=0.0,            # We don't have volume data for partial candle
            timeframe=timeframe,
        )

        # Calculate VWAP deviation
        vwap_deviation = None
        if last_snapshot.vwap is not None:
            vwap_deviation = self._vwap[product_id][timeframe].get_deviation(live_price)

        # Calculate live RSI using the partial candle
        # This is TradingView's approach: use a cloned RSI calculator and update with live price
        live_rsi = last_snapshot.rsi  # Default to last candle's RSI

        if timeframe in self._live_rsi.get(product_id, {}):
            live_rsi_calc = self._live_rsi[product_id][timeframe]

            # Update the live RSI calculator with the partial candle
            # This simulates what RSI would be if the candle closed right now
            live_rsi_value = live_rsi_calc.update(partial_candle)

            if live_rsi_value is not None:
                live_rsi = live_rsi_value
                logger.debug(
                    f"Live RSI update: {product_id} {timeframe.value} - "
                    f"Official RSI: {last_snapshot.rsi:.1f}, Live RSI: {live_rsi:.1f}, "
                    f"Price: ${live_price:,.2f}"
                )

            # Important: Re-sync live calculator for next update
            # We need to reset it back to the official state after this calculation
            # because we'll get multiple live updates before the next candle close
            self._live_rsi[product_id][timeframe] = self._rsi[product_id][timeframe].clone()

        # ADX and trend also stay from last candle (need multiple bars to calculate)
        # But we can update ATR percentage with live price
        atr_pct = None
        if last_snapshot.atr is not None and live_price > 0:
            atr_pct = (last_snapshot.atr / live_price) * 100

        # Create updated snapshot with live price data
        snapshot = IndicatorSnapshot(
            product_id=product_id,
            timeframe=timeframe,
            timestamp=time.time(),
            vwap=last_snapshot.vwap,
            vwap_deviation=vwap_deviation,  # Live VWAP deviation
            adx=last_snapshot.adx,
            plus_di=last_snapshot.plus_di,
            minus_di=last_snapshot.minus_di,
            trend_direction=last_snapshot.trend_direction,
            atr=last_snapshot.atr,
            atr_pct=atr_pct,  # Live ATR percentage
            volatility=last_snapshot.volatility,
            regime=last_snapshot.regime,
            rsi=live_rsi,  # Live RSI (approximated for now)
            macd_line=last_snapshot.macd_line,
            macd_signal=last_snapshot.macd_signal,
            macd_histogram=last_snapshot.macd_histogram,
            bb_upper=last_snapshot.bb_upper,
            bb_middle=last_snapshot.bb_middle,
            bb_lower=last_snapshot.bb_lower,
        )

        # DON'T update stored snapshot - keep the official one from candle close
        # This way the next candle close still has clean state
        # We just emit the live update to the UI

        # Emit update (but don't store it)
        if self.on_indicator_update:
            await self.on_indicator_update(snapshot)

        return snapshot

    def _determine_regime(
        self,
        product_id: str,
        timeframe: Timeframe,
        adx: Optional[float],
        volatility: Optional[float]
    ) -> str:
        if adx is None:
            return "UNKNOWN"

        # High ADX = trending
        if adx >= self.config.regime_change_adx_threshold:
            return "TRENDING"

        # High volatility but low ADX = volatile ranging
        if volatility is not None and volatility > 50:
            return "VOLATILE"

        # Low ADX = ranging
        return "RANGING"

    async def _check_alerts(
        self,
        product_id: str,
        timeframe: Timeframe,
        snapshot: IndicatorSnapshot,
        candle: Candle
    ):
        if not self.on_alert:
            return

        # Get previous snapshot for this timeframe
        prev_snapshot = self._last_snapshots.get(product_id, {}).get(timeframe)

        # Check regime change
        if prev_snapshot and prev_snapshot.regime != snapshot.regime:
            if snapshot.regime != "UNKNOWN":
                alert = Alert(
                    alert_type="regime_change",
                    message=f"Market regime changed from {prev_snapshot.regime} to {snapshot.regime} on {timeframe.value}",
                    product_id=product_id,
                    timestamp=time.time(),
                    data={
                        "timeframe": timeframe.value,
                        "previous_regime": prev_snapshot.regime,
                        "new_regime": snapshot.regime,
                        "adx": snapshot.adx,
                    },
                    severity="WARNING" if snapshot.regime == "VOLATILE" else "INFO",
                )
                await self.on_alert(alert)

        # Check volume spike
        volume_history = self._volume_history.get(product_id, {}).get(timeframe, deque())
        if len(volume_history) >= 5 and candle.volume > 0:
            avg_volume = sum(list(volume_history)[:-1]) / (len(volume_history) - 1)
            if avg_volume > 0:
                volume_ratio = candle.volume / avg_volume
                if volume_ratio >= self.config.volume_spike_threshold:
                    alert = Alert(
                        alert_type="volume_spike",
                        message=f"Volume spike on {timeframe.value}: {volume_ratio:.1f}x average",
                        product_id=product_id,
                        timestamp=time.time(),
                        data={
                            "timeframe": timeframe.value,
                            "volume": candle.volume,
                            "avg_volume": avg_volume,
                            "ratio": volume_ratio,
                        },
                        severity="WARNING",
                    )
                    await self.on_alert(alert)

    def get_snapshot(self, product_id: str, timeframe: Optional[Timeframe] = None) -> Optional[IndicatorSnapshot]:
        if product_id not in self._last_snapshots:
            return None

        if timeframe:
            return self._last_snapshots[product_id].get(timeframe)

        # Return latest snapshot (any timeframe)
        snapshots = self._last_snapshots[product_id].values()
        return max(snapshots, key=lambda s: s.timestamp) if snapshots else None

    def get_vwap(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        calc = self._vwap.get(product_id, {}).get(timeframe)
        return calc.current_vwap if calc else None

    def get_adx(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        calc = self._adx.get(product_id, {}).get(timeframe)
        return calc.current_adx if calc else None

    def get_atr(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        calc = self._atr.get(product_id, {}).get(timeframe)
        return calc.current_atr if calc else None

    def get_volatility(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        calc = self._volatility.get(product_id, {}).get(timeframe)
        return calc.current_volatility if calc else None

    def get_rsi(self, product_id: str, timeframe: Timeframe) -> Optional[float]:
        calc = self._rsi.get(product_id, {}).get(timeframe)
        return calc.current_rsi if calc else None

    def reset(self, product_id: Optional[str] = None, timeframe: Optional[Timeframe] = None):
        calc_dicts = [self._vwap, self._adx, self._atr, self._volatility,
                      self._rsi, self._macd, self._bb]
        state_dicts = [self._last_snapshots, self._last_prices, self._volume_history]

        if product_id and timeframe:
            for calc_dict in calc_dicts:
                if product_id in calc_dict and timeframe in calc_dict[product_id]:
                    calc_dict[product_id][timeframe].reset()
            for state_dict in state_dicts:
                if product_id in state_dict:
                    state_dict[product_id].pop(timeframe, None)
        elif product_id:
            for calc_dict in calc_dicts:
                if product_id in calc_dict:
                    for calc in calc_dict[product_id].values():
                        calc.reset()
            for state_dict in state_dicts:
                if product_id in state_dict:
                    state_dict[product_id].clear()
        else:
            for calc_dict in calc_dicts:
                for product_calcs in calc_dict.values():
                    for calc in product_calcs.values():
                        calc.reset()
            for state_dict in state_dicts:
                state_dict.clear()
