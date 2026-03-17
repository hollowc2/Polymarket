# Conventions

## Code Style
- Ruff for linting and formatting (default config)
- ty for type checking — avoid `Any`, use proper types
- Type hints on all function signatures
- PEP 695 `type` keyword for type aliases (Python 3.12+)

## Package Boundaries
- **core**: Protocol definitions, config, plugin registry. No business logic. No external API calls.
- **data**: Data fetching and storage only. No strategy logic.
- **indicators**: Pure numpy/pandas computations. No side effects. No API calls.
- **strategies**: Signal generation via Strategy Protocol. May use indicators. No direct API calls.
- **backtest**: Engine and metrics. Accepts Strategy Protocol objects or callables.
- **executor**: API clients and execution. No strategy logic. Depends only on core.
- **scripts/**: Thin CLI wrappers. Orchestration only.

## Protocols
- New strategies MUST conform to `Strategy` Protocol (name, description, timeframe, evaluate, default_params, param_grid)
- Event-driven strategies (e.g. copytrade) set `kind = "event_driven"` and raise `NotImplementedError` in `evaluate()`
- New indicators MUST conform to `Indicator` Protocol (name, compute)
- New data feeds MUST conform to `DataFeed` Protocol (start, stop, subscribe, unsubscribe, on_tick, on_reconnect, is_connected)
- Filters (e.g. SelectiveFilter) are NOT strategies — don't register them as strategy entry points

## Naming
- Market-related: `condition_id`, `token_id`, `slug`
- Strategy signals: use explicit types/dataclasses, not raw dicts
- Config: ALL_CAPS env vars → snake_case Python attributes in `Config` class
- Packages: `polymarket_algo.*` namespace

## Error Handling
- Graceful degradation: if a market fetch fails, skip and continue
- WebSocket always has REST fallback
- Circuit breaker wraps external API calls
- Never crash on a single failed trade

## Concurrency
- Use `threading.Event` (not `asyncio.Event`) for cross-thread synchronization
- Protect shared mutable state with `threading.Lock`
- Copy callback lists under lock before iterating

## Git
- Conventional commits: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`
- Pre-commit: ruff lint + format
- Pre-push: ty typecheck
- Hooks managed by prek
