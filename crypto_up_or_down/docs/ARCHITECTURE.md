# Architecture

## Overview
Composable trading toolkit for Polymarket prediction markets. Monorepo of independently installable packages connected via Protocol-based interfaces.

## Package Dependency Graph

```
core (protocols, config, plugin registry)
 ├── data        → Binance fetcher + storage
 ├── indicators  → pure numpy/pandas computations
 ├── strategies  → Strategy protocol implementations (depends on indicators)
 ├── backtest    → engine + metrics (depends on core)
 └── executor    → Polymarket client, WebSocket, trader (depends on core)
```

No circular dependencies. Each package declares its own deps in `pyproject.toml`.

## Core Protocols (`packages/core/`)

### Strategy Protocol
```python
class Strategy(Protocol):
    name: str
    description: str
    timeframe: str
    default_params: dict
    param_grid: dict
    def evaluate(self, candles: pd.DataFrame, **params) -> pd.Series | pd.DataFrame: ...
```

Implementations: `StreakReversalStrategy`, `CandleDirectionStrategy`, `CopytradeStrategy` (event-driven, kind="event_driven").

### Indicator Protocol
```python
class Indicator(Protocol):
    name: str
    def compute(self, data: pd.Series, **params) -> pd.Series | pd.DataFrame: ...
```

Implementations: EMA, SMA, RSI, MACD, Bollinger Bands. Both raw functions and Protocol-conforming wrapper classes are exported.

### DataFeed Protocol
```python
@dataclass
class PriceTick:
    symbol: str
    price: float
    timestamp: float
    size: float | None = None
    side: str | None = None   # "buy" | "sell"
    source: str = ""

class DataFeed(Protocol):
    name: str
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def subscribe(self, symbol: str, **kwargs) -> None: ...
    def unsubscribe(self, symbol: str) -> None: ...
    def on_tick(self, callback: Callable[[PriceTick], None]) -> None: ...
    def on_reconnect(self, callback: Callable[[], None]) -> None: ...
    def is_connected(self) -> bool: ...
```

Built-in: `PolymarketDataFeed` (wraps `PolymarketWebSocket`, emits ticks on trades + orderbook mid changes). Future: Binance WS, Chainlink oracle feeds.

### Plugin Registry
`PluginRegistry` unifies discovery from:
- Python entry points (`polymarket_algo.strategies`, `polymarket_algo.indicators`)
- Local drop-in plugins (`~/.polymarket-algo/plugins/*.py`)

## Executor Layer (`packages/executor/`)

### Client (`client.py`)
- `PolymarketClient` — REST client for Gamma (market discovery) and CLOB (orderbook/prices) APIs
- `Market` — market data model
- `DelayImpactModel` — non-linear delay impact calculator for copytrade

### WebSocket (`ws.py`)
- `PolymarketWebSocket` — real-time orderbook + trade feed (~100ms latency). Exposes `on_trade` callback and `on_mid_change` for orderbook mid-price updates.
- `UserWebSocket` — authenticated feed for order lifecycle events (MATCHED/CONFIRMED/FAILED)
- Both use exponential backoff reconnection and `threading.Event` for cross-thread sync.

### Trader (`trader.py`)
- `PaperTrader` — simulation mode, logs trades to JSON
- `LiveTrader` — submits FOK orders via CLOB API, quarter-Kelly sizing
- `TradingState` — tracks bankroll, positions, daily limits

### Resilience (`resilience.py`)
- `CircuitBreaker` — prevents cascading failures
- `RateLimiter` — respects API rate limits
- `HealthCheck` — system health monitoring

### DataFeed Adapter (`feed.py`)
- `PolymarketDataFeed` — thin wrapper conforming to `DataFeed` protocol, emits `PriceTick` on trades and orderbook mid changes

## Data Flow

### Streak Strategy
1. Gamma API → fetch recent resolved BTC 5-min markets
2. Detect streak of N consecutive same outcomes
3. ~30s before next window → place reversal bet at ~50¢
4. Market resolves → track P&L

### Copytrade
1. Poll target wallet activity (data API, 1.5s interval)
2. Detect new BTC 5-min position
3. Fetch orderbook (WebSocket or REST fallback)
4. Place matching FOK order via `LiveTrader`
5. Track execution quality (delay, spread, slippage)

### Backtesting
1. `packages/data/` fetches historical OHLCV from Binance
2. `packages/backtest/engine.py` runs strategy `.evaluate()` over candles
3. Parameter sweep tests all combinations from `param_grid`
4. Walk-forward split validates out-of-sample performance

## External APIs
| API | Auth | Purpose |
|-----|------|---------|
| Gamma API | None | Market discovery (slugs, outcomes) |
| CLOB API (REST) | None / API key | Orderbook, prices, order placement |
| CLOB API (WS) | None | Real-time orderbook |
| Polymarket Data API | None | Wallet activity monitoring |
| Polygonscan | API key | On-chain wallet data |
| Binance | None | Historical OHLCV data for backtesting |

## Market Mechanics
- Slug pattern: `btc-updown-5m-{unix_ts}` every 300s
- Two outcomes per market: Up / Down (binary)
- Trading requires Polygon wallet + EIP-712 derived API credentials
- ~5% Polymarket fee on winnings
