from .adx import adx
from .bollinger import bollinger_bands
from .ema import ema
from .hl_orderflow import HLOrderFlowIndicator as HLOrderFlowIndicator
from .hl_orderflow import hl_orderflow as hl_orderflow
from .hl_orderflow import hl_orderflow_signal as hl_orderflow_signal
from .macd import macd
from .rsi import rsi
from .sma import sma


class ADXIndicator:
    name = "adx"

    def compute(self, high, low, close, **params):
        return adx(high, low, close, period=int(params.get("period", 14)))


class EMAIndicator:
    name = "ema"

    def compute(self, series, **params):
        return ema(series, period=int(params.get("period", 20)))


class SMAIndicator:
    name = "sma"

    def compute(self, series, **params):
        return sma(series, period=int(params.get("period", 20)))


class RSIIndicator:
    name = "rsi"

    def compute(self, series, **params):
        return rsi(series, period=int(params.get("period", 14)))


class MACDIndicator:
    name = "macd"

    def compute(self, series, **params):
        return macd(
            series,
            int(params.get("fast_period", 12)),
            int(params.get("slow_period", 26)),
            int(params.get("signal_period", 9)),
        )


class BollingerIndicator:
    name = "bollinger"

    def compute(self, series, **params):
        return bollinger_bands(series, int(params.get("period", 20)), float(params.get("std_dev", 2.0)))
