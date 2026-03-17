from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd


class Indicator(Protocol):
    name: str

    def compute(self, series: pd.Series, **params: Any) -> pd.Series | pd.DataFrame: ...


class Strategy(Protocol):
    name: str
    description: str
    timeframe: str

    def evaluate(self, candles: pd.DataFrame, **params: Any) -> pd.DataFrame: ...
    @property
    def default_params(self) -> dict[str, Any]: ...
    @property
    def param_grid(self) -> dict[str, list[Any]]: ...


@dataclass
class PriceTick:
    """Normalized price update from any data feed."""

    symbol: str
    price: float
    timestamp: float
    size: float | None = None
    side: str | None = None  # "buy" | "sell"
    source: str = ""


@runtime_checkable
class DataFeed(Protocol):
    """Protocol for market data feeds (Polymarket, Binance, Chainlink, etc.)."""

    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def subscribe(self, symbol: str, **kwargs: Any) -> None: ...
    def unsubscribe(self, symbol: str) -> None: ...
    def on_tick(self, callback: Callable[[PriceTick], None]) -> None: ...
    def on_reconnect(self, callback: Callable[[], None]) -> None: ...
    def is_connected(self) -> bool: ...
