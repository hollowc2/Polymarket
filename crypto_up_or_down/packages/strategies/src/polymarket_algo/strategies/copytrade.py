class CopytradeStrategy:
    """Wallet activity driven strategy.

    This strategy is event-driven and consumes live wallet activity streams,
    so it cannot be backtested on candle-only historical data.
    """

    name = "copytrade"
    description = "Wallet activity driven copytrade strategy"
    timeframe = "5m"
    kind = "event_driven"

    @property
    def default_params(self):
        return {"poll_interval": 1.5}

    @property
    def param_grid(self):
        return {"poll_interval": [1.0, 1.5, 2.0]}

    def evaluate(self, candles, **params):
        raise NotImplementedError("Copytrade strategy is event-driven")
