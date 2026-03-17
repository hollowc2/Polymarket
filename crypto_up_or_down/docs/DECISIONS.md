# Architecture Decision Records

## ADR-001: Trigger Length = 4
**Context:** Need to pick how many consecutive same outcomes before betting reversal.
**Decision:** Trigger=4. Balances trade frequency (~51 signals in sample) with win rate (~67%).
**Alternatives:** Trigger=5 has higher win rate (82%) but very few signals (17 in sample).

## ADR-002: Quarter-Kelly Sizing
**Context:** Kelly criterion suggests optimal bet size but is aggressive.
**Decision:** Use quarter-Kelly for conservative bankroll management. Reduces variance at cost of slower growth.

## ADR-003: FOK Orders for Copytrade
**Context:** Copytrade needs guaranteed fills — partial fills create position management complexity.
**Decision:** Fill-or-Kill (FOK) orders. Either fully filled or not at all.

## ADR-004: Hybrid WS + REST for Copytrade Detection
**Context:** Pure WebSocket for trade detection was unreliable. Pure REST has 5s+ latency.
**Decision:** Hybrid approach — WebSocket for orderbook data, fast REST polling (1.5s) for wallet activity. Achieves ~1.5-2s total detection latency.

## ADR-005: Monorepo with uv Workspaces
**Context:** Flat `src/` layout became unwieldy. Needed independently testable, composable packages.
**Decision:** Monorepo with `packages/` directory. Each package (core, data, indicators, strategies, backtest, executor) has its own `pyproject.toml` and is independently installable. uv workspaces manage inter-package deps.
**Supersedes:** ADR-005 (old src/ layout).

## ADR-006: Nix + uv Dual Tooling
**Context:** Need reproducible dev environment. Nix provides system tools but not all contributors use Nix.
**Decision:** Nix flake provides python, uv, prek, ruff, ty. Dev tools (ruff, ty, pytest) also in uv dev deps for non-Nix users. Git hook entries use bare command names — whichever is on PATH works.

## ADR-007: Protocol-Based Plugin System
**Context:** Need extensible strategies and indicators without inheritance coupling.
**Decision:** Python `Protocol` (structural typing) for Strategy, Indicator, and DataFeed interfaces. Discovery via entry points + local drop-in plugins. `PluginRegistry` unifies both.

## ADR-008: DataFeed Protocol for Multi-Source Feeds
**Context:** Polymarket WebSocket is the only data source. Want to add Binance WS and Chainlink oracle feeds for faster price discovery and lower latency.
**Decision:** `DataFeed` protocol with `PriceTick` dataclass. Each feed implements subscribe/unsubscribe/on_tick/on_reconnect. Built-in `PolymarketDataFeed` wraps existing WS. Future feeds (Binance, Chainlink) implement the same interface. Trader subscribes to any `DataFeed` without knowing the source.

## ADR-009: Threading Safety in WebSocket Layer
**Context:** WebSocket classes used `asyncio.Event` for cross-thread sync — undefined behavior, breaks on free-threaded Python 3.13+.
**Decision:** Use `threading.Event` for cross-thread signaling. Protect shared mutable state (`_subscribed_markets`, `_trade_callbacks`) with `threading.Lock`. Copy callback lists under lock before iterating.
