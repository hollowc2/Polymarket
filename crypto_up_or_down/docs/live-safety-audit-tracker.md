# Live Safety Audit Tracker

Created: 2026-06-29

Purpose: resumable implementation tracker for the live-trading safety audit. Update this file after each completed item so work can stop and resume without losing context.

Status legend:

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked or needs decision

## Current Focus

- `[x]` Live-safety tracker complete.

## Phase 0 - Audit Review And Tracking

- `[x]` Create this tracking document before implementation.
- `[x]` Spawn read-only agents to validate the highest-risk execution finding and the failing baseline tests.
- `[x]` Re-check audit line references against current code before editing the baseline and FOK areas.

## Phase 1 - Clean Baseline

- `[x]` Fix `tests/test_three_bar_momo.py` failures or update the strategy contract intentionally.
- `[x]` Remove Ruff failure in `packages/strategies/src/polymarket_algo/strategies/deribit_skew.py`.
- `[x]` Verify targeted Ruff and ThreeBarMoMo tests.

## Phase 2 - FOK Order Result Safety

- `[x]` Ensure live FOK submission returns no `Trade` unless exchange status confirms filled.
- `[x]` Record failed/rejected/unknown attempts separately from filled trades.
- `[x]` Add focused tests for exception, rejected/cancelled, and unknown status behavior.
- `[x]` Verify targeted executor tests.

## Phase 3 - Durable Order Intent Ledger

- `[x]` Add append-only order intent/result ledger or minimal atomic JSON ledger.
- `[x]` Persist intent before submit and exchange order id immediately after submit.
- `[x]` Add idempotency key per strategy, market, direction, and window.
- `[x]` Add tests for duplicate submit and corrupt ledger/state handling.

## Phase 4 - Startup Reconciliation

- `[x]` Reconcile local ledger/state with open/recent CLOB orders at startup.
- `[x]` Refuse new live orders when reconciliation is stale, failing, or incomplete.
- `[x]` Keep failed cancel attempts in unresolved state until confirmed cancelled/filled.
- `[x]` Add restart/open-order/cancel-failure tests.

## Phase 5 - Central Live Risk Gates

- `[x]` Add central live risk guard inside execution, not only script-level checks.
- `[x]` Enforce max open orders, max per-market exposure, max per-strategy exposure, max notional, max order price, stale quote limit, and manual kill switch.
- `[x]` Add tests for each hard rejection.

## Phase 6 - Market Data Freshness And Typed Errors

- `[x]` Add typed market/book snapshots with `fetched_at`, source, stale/error status, and diagnostics.
- `[x]` Fail closed on stale, malformed, missing, or one-sided execution data where live trading requires it.
- `[x]` Add tests for stale data, API errors, empty books, and one-sided books.

## Phase 7 - Live Config Safety

- `[x]` Add explicit app mode and live confirmation/account validation before live trader startup.
- `[x]` Reject missing or mismatched live-only env vars.
- `[x]` Add startup validation tests.

## Phase 8 - Observability

- `[x]` Add structured order/risk/reconciliation/market-data logs.
- `[x]` Add Prometheus metrics for order status, unknowns, open orders, quote age, API errors, reconciliation, cancel failures, kill switch, and exposure.
- `[x]` Add or update alerts after metrics exist.

## Verification Log

- 2026-06-29: Tracker created. Agents spawned for FOK execution and ThreeBarMoMo/Ruff validation.
- 2026-06-29: Phase 1 complete. Fixed short-history volume spike gate in `three_bar_momo.py`, removed unused `numpy` import from `deribit_skew.py`, verified `uv run pytest tests/test_three_bar_momo.py -q` and targeted Ruff pass with `/tmp` caches.
- 2026-06-29: Phase 2 FOK trade-safety complete. `LiveTrader.place_bet()` now returns `None` for rejected/malformed submit responses, cancelled FOK, unknown status, and submit exceptions; only confirmed fills return `Trade(order_status="filled")`. Added `tests/test_live_trader_fok.py`. Verified `uv run pytest tests/test_live_trader_fok.py tests/test_three_bar_momo.py -q` and targeted Ruff pass with `/tmp` caches. Durable rejected/unknown attempt recording remains open under Phase 3.
- 2026-06-29: Phase 3 first slice complete. Added `JsonOrderLedger` at `packages/executor/src/polymarket_algo/executor/order_ledger.py` with `ORDER_LEDGER_FILE`; live FOK now records intent before submit, submitted order id after submit, and filled/cancelled/rejected/unknown/failed events. Intent persistence failure fails closed before submit. Verified focused pytest and Ruff pass. Remaining Phase 3 work: duplicate/idempotency enforcement and corrupt ledger/state handling.
- 2026-06-29: Phase 7 complete. `LiveTrader` now requires `APP_MODE=live` and `LIVE_CONFIRM=crypto_up_or_down:<wallet>` before client initialization, using `WALLET_ADDRESS` for EOA or `FUNDER_ADDRESS` for proxy wallets. Added `tests/test_live_config_safety.py`. Verified focused pytest and Ruff pass.
- 2026-06-29: Phase 5 first slice complete. Added central live FOK risk guard for `LIVE_KILL_SWITCH`, `LIVE_KILL_SWITCH_FILE`, `MAX_LIVE_ORDER_USD`, and `MAX_LIVE_ORDER_PRICE`, all before ledger write or CLOB submit. Added focused tests. Remaining Phase 5 work: ledger/account-backed open-order, per-market, per-strategy, aggregate exposure, and stale quote enforcement.
- 2026-06-29: Broad verification passed. `uv run pytest -q` returned `111 passed`; `uv run ruff check packages tests scripts/bots/impulse_momentum_bot.py` returned `All checks passed!` with `UV_CACHE_DIR=/tmp/uv-cache RUFF_CACHE_DIR=/tmp/ruff-cache`.
- 2026-06-29: Commit checkpoint prepared on branch `feat/impulse-momentum-bot` for the first live-safety audit implementation slice. Scope: tracker, baseline test fixes, live FOK fail-closed semantics, JSONL order ledger, live confirmation gate, and first central live risk guards.
- 2026-06-29: Phase 3 duplicate-idempotency slice complete. `JsonOrderLedger.has_intent()` now rejects duplicate strategy/market/direction/window intents and fails closed on corrupt JSONL before signing/submitting live FOK orders. Added duplicate-submit and corrupt-ledger tests. Verified `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_live_trader_fok.py -q`, `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`, and targeted Ruff pass with `/tmp` caches.
- 2026-06-29: Phase 4 and remaining concrete Phase 5 central-risk slice complete. `LiveTrader` now reconciles open/live CLOB orders at startup, records startup cancel success/failure in the order ledger, and refuses live FOK submits when reconciliation is incomplete. `JsonOrderLedger.risk_snapshot()` now backs live guards for max open orders, per-market exposure, per-strategy exposure, and total notional; stale quote rejection uses `precomputed_execution.fetched_at_ms` or `timestamp_ms` when `MAX_LIVE_QUOTE_AGE_SECONDS` is enabled. Verified `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_live_trader_fok.py -q` and `UV_CACHE_DIR=/tmp/uv-cache RUFF_CACHE_DIR=/tmp/ruff-cache uv run ruff check packages/executor/src/polymarket_algo/executor/trader.py packages/executor/src/polymarket_algo/executor/order_ledger.py packages/core/src/polymarket_algo/core/config.py tests/test_live_trader_fok.py`.
- 2026-06-29: Phase 8 exporter metrics slice complete. `scripts/grafana_exporter.py` now exposes Prometheus gauges for latest order ledger status, unknown orders, open/unresolved orders, ledger read health, state/ledger cancel failures, kill switch env/file state, pending/open exposure, reconciliation stale/failed state, impulse quote age, impulse API errors, and health-file read state. `scripts/bots/impulse_momentum_bot.py` writes `impulse-momentum-health.json` beside its trade state as the quote/API metric source. Reused existing exporter logging for unreadable ledger/reconciliation/health files. Verified focused pytest and Ruff passes.
- 2026-06-29: Phase 6 impulse-bot slice complete. `BookSnapshot` now carries executable depth plus `fetched_at`, `source`, stale/error status, and diagnostics; signal/entry quote fetches and paper exits fail closed on stale, API-error, empty, malformed, or one-sided book data before execution pricing or paper settlement. Verified `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_impulse_momentum.py -q` and `UV_CACHE_DIR=/tmp/uv-cache RUFF_CACHE_DIR=/tmp/ruff-cache uv run ruff check scripts/bots/impulse_momentum_bot.py tests/test_impulse_momentum.py`.
- 2026-06-29: Phase 8 alert slice complete in `/opt/monitoring/prometheus-alerts/cryptoupdown.yml`. Added alerts for unreadable order ledger, unresolved orders, cancel failures, active kill switch, unreadable health state, slow quote fetches, and API error increases. Verified `docker exec butterfly_prometheus promtool check rules /etc/prometheus/alerts/cryptoupdown.yml`, reloaded Prometheus, and confirmed the new CryptoUpDown rules are loaded via `/api/v1/rules`.
- 2026-06-29: Integrated verification passed after all agent slices and local fixes: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` returned `134 passed`; `UV_CACHE_DIR=/tmp/uv-cache RUFF_CACHE_DIR=/tmp/ruff-cache uv run ruff check packages tests scripts/bots/impulse_momentum_bot.py scripts/grafana_exporter.py` returned `All checks passed!`.
- 2026-06-29: Runtime deploy check complete. Rebuilt/restarted `impulse-momentum-bot`, restarted bind-mounted `grafana-exporter`, confirmed impulse bot startup logs, and confirmed exporter now exposes `polymarket_order_ledger_read_success`, `polymarket_live_kill_switch_active`, `polymarket_api_errors_total`, and `polymarket_health_read_success`; `polymarket_quote_age_sec` exists and will emit a sample after the next quote fetch.
- 2026-06-29: Phase 8 structured logging slice complete. `LiveTrader` now emits compact JSON `live_execution` lines for live order intents, submitted/filled/cancelled/rejected/unknown/failed order lifecycle events, risk/validation/idempotency/ledger rejections, and startup reconciliation outcomes while preserving the durable JSONL order ledger. Verified `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_live_trader_fok.py -q` and `UV_CACHE_DIR=/tmp/uv-cache RUFF_CACHE_DIR=/tmp/ruff-cache uv run ruff check packages/executor/src/polymarket_algo/executor/trader.py tests/test_live_trader_fok.py`.

## Resume Notes

- Start each resumed session by reading this tracker and `git status --short`.
- Do not mark an item complete until the relevant test/lint command is run or the skipped verification reason is recorded here.
- Keep live trading disabled by default. Paper mode remains the only safe default until all must-fix phases are complete.
