FROM python:3.13-slim

# Install uv from official image (pinned version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- Phase 1: install deps (cached unless lock/pyproject changes) ---
# Copy workspace manifests first — if these don't change, the dep install layer is reused
COPY pyproject.toml uv.lock .python-version ./

COPY packages/core/pyproject.toml            packages/core/pyproject.toml
COPY packages/data/pyproject.toml            packages/data/pyproject.toml
COPY packages/indicators/pyproject.toml      packages/indicators/pyproject.toml
COPY packages/strategies/pyproject.toml      packages/strategies/pyproject.toml
COPY packages/backtest/pyproject.toml        packages/backtest/pyproject.toml
COPY packages/executor/pyproject.toml        packages/executor/pyproject.toml
# Stub src dirs so uv can resolve workspace members before source is copied
RUN mkdir -p \
    packages/core/src/polymarket_algo/core \
    packages/data/src/polymarket_algo/data \
    packages/indicators/src/polymarket_algo/indicators \
    packages/strategies/src/polymarket_algo/strategies \
    packages/backtest/src/polymarket_algo/backtest \
    packages/executor/src/polymarket_algo/executor

# Install all third-party deps from lockfile (no dev deps)
# --frozen: never re-resolve, fail if lockfile is stale
# --no-install-project: skip root package (it has no src/)
RUN uv sync --all-packages --frozen --no-install-project --no-dev

# --- Phase 2: copy source (invalidated on code changes, deps stay cached) ---
COPY packages/ packages/
COPY scripts/  scripts/

# Re-sync to install workspace members now that source exists (fast: deps already cached)
RUN uv sync --all-packages --frozen --no-install-project --no-dev

# State dir for persistent files (bind-mounted at runtime via docker-compose)
RUN mkdir -p /app/state

# Default: streak bot in paper mode. Overridden per-service in docker-compose.yml.
CMD ["uv", "run", "python", "scripts/bots/streak_bot.py", "--paper"]
