# VPS Deployment Plan — Polymarket Auto Trader

## Context

The bot runs locally but needs to execute 24/7. The Hostinger VPS (Ubuntu 24.04, Docker pre-installed, 2 CPU / 8 GB RAM / 100 GB disk) accessed via Tailscale SSH is the target.

**Workflow:** develop locally → `git push` → SSH to VPS → `git pull` → `docker compose up -d --build`

Starting in paper trade mode. Designed to support multiple bots as they are added.

---

## Files to Create

### `Dockerfile`

Two-phase uv install: copy metadata first (cached layer), then copy source (fast cache invalidation). uv binary copied from the official image.

```dockerfile
FROM python:3.13-slim

# Install uv from official image (pinned version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

# uv venv path + PATH so 'uv run' works
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- Phase 1: install deps (cached unless lock/pyproject changes) ---
COPY pyproject.toml uv.lock .python-version ./

COPY packages/core/pyproject.toml           packages/core/pyproject.toml
COPY packages/data/pyproject.toml           packages/data/pyproject.toml
COPY packages/indicators/pyproject.toml     packages/indicators/pyproject.toml
COPY packages/strategies/pyproject.toml     packages/strategies/pyproject.toml
COPY packages/backtest/pyproject.toml       packages/backtest/pyproject.toml
COPY packages/executor/pyproject.toml       packages/executor/pyproject.toml
COPY examples/custom_strategy/pyproject.toml examples/custom_strategy/pyproject.toml

# Stub src dirs so uv can resolve workspace members before source is copied
RUN mkdir -p \
    packages/core/src/polymarket_algo/core \
    packages/data/src/polymarket_algo/data \
    packages/indicators/src/polymarket_algo/indicators \
    packages/strategies/src/polymarket_algo/strategies \
    packages/backtest/src/polymarket_algo/backtest \
    packages/executor/src/polymarket_algo/executor \
    examples/custom_strategy/src

# Install all third-party deps from lockfile (no dev deps)
RUN uv sync --all-packages --frozen --no-install-project --no-dev

# --- Phase 2: copy source (invalidated on code changes, deps stay cached) ---
COPY packages/ packages/
COPY scripts/  scripts/
COPY examples/ examples/

# Re-sync to install workspace members now that source exists (fast: deps cached)
RUN uv sync --all-packages --frozen --no-install-project --no-dev

# State dir for persistent files (mounted as a volume at runtime)
RUN mkdir -p /app/state

# Default: streak bot in paper mode. Overridden per-service in docker-compose.yml.
CMD ["uv", "run", "python", "scripts/streak_bot.py", "--paper"]
```

---

### `docker-compose.yml`

YAML anchors (`&bot-common`) share config across services — adding a new bot is one uncommented block.

```yaml
x-bot-common: &bot-common
  build:
    context: .
    dockerfile: Dockerfile
  restart: unless-stopped
  env_file:
    - .env                                   # SCP'd to VPS, never in git/image
  volumes:
    - /opt/polymarket/state:/app/state       # trades.json, bot.log, history files
    - /opt/polymarket/data:/app/data         # Binance OHLCV parquet files
  logging:
    driver: "json-file"
    options:
      max-size: "10m"
      max-file: "5"

services:

  streak-bot:
    <<: *bot-common
    container_name: polymarket-streak-bot
    command: ["uv", "run", "python", "scripts/streak_bot.py", "--paper"]
    environment:
      PAPER_TRADE: "true"                    # Safety override — remove to go live
      TRADES_FILE: "/app/state/trades.json"
      HISTORY_FILE: "/app/state/trade_history_full.json"
      LOG_FILE: "/app/state/bot.log"

  # --- Copy Bot (uncomment to enable) ---
  # copy-bot:
  #   <<: *bot-common
  #   container_name: polymarket-copy-bot
  #   command: ["uv", "run", "python", "scripts/copybot.py"]
  #   environment:
  #     PAPER_TRADE: "true"
  #     COPY_WALLETS: "0xTargetWalletHere"
  #     TRADES_FILE: "/app/state/copybot-trades.json"
  #     HISTORY_FILE: "/app/state/copybot-history.json"
  #     LOG_FILE: "/app/state/copybot.log"

  # --- New Bot Template ---
  # my-new-bot:
  #   <<: *bot-common
  #   container_name: polymarket-my-new-bot
  #   command: ["uv", "run", "python", "scripts/my_bot.py", "--paper"]
  #   environment:
  #     PAPER_TRADE: "true"
  #     TRADES_FILE: "/app/state/my-bot-trades.json"
  #     HISTORY_FILE: "/app/state/my-bot-history.json"
  #     LOG_FILE: "/app/state/my-bot.log"
```

---

### `.dockerignore`

```
__pycache__/
*.pyc
.venv/
.direnv/
result
flake.nix
flake.lock
.env
.envrc
trades.json
trade_history_full.json
data/
backtest_results/
.git/
.github/
.vscode/
pyrightconfig.json
docs/
bot.py
copybot.py
copybot_v2.py
backtest_engine.py
src/
tests/
```

---

## Files to Modify

### `packages/core/src/polymarket_algo/core/config.py` (lines 48–49)

`LOG_FILE` and `TRADES_FILE` are currently hardcoded strings. Change to env-configurable, and add `HISTORY_FILE`:

```python
# Before:
LOG_FILE: str = "bot.log"
TRADES_FILE: str = "trades.json"

# After:
LOG_FILE: str = os.getenv("LOG_FILE", "bot.log")
TRADES_FILE: str = os.getenv("TRADES_FILE", "trades.json")
HISTORY_FILE: str = os.getenv("HISTORY_FILE", "trade_history_full.json")
```

### `packages/executor/src/polymarket_algo/executor/trader.py` (lines 555, 591, 901, 926, 1035)

Five occurrences of hardcoded `"trade_history_full.json"`. Replace all with `Config.HISTORY_FILE`:

```python
# Before (×5):
history_file = "trade_history_full.json"

# After (×5):
history_file = Config.HISTORY_FILE
```

Ensure `Config` is imported at the top of `trader.py` (check if already present).

### `.env.example`

Add a deployment section at the bottom:

```bash
# File paths (set these when running in Docker)
# TRADES_FILE=/app/state/trades.json
# HISTORY_FILE=/app/state/trade_history_full.json
# LOG_FILE=/app/state/bot.log
```

---

## VPS First-Time Setup

Run these commands on the VPS via SSH (Tailscale):

```bash
# 1. Verify Docker is ready
docker --version && docker compose version

# 2. Create persistent host directories
sudo mkdir -p /opt/polymarket/state /opt/polymarket/data
sudo chown -R $USER:$USER /opt/polymarket

# 3. Clone repo (SSH key already configured on VPS)
git clone git@github.com:YOUR_USERNAME/polymarket_auto_trader.git /opt/polymarket/app
cd /opt/polymarket/app

# 4. Copy .env from LOCAL machine (run this on your desktop, not the VPS):
scp .env YOUR_VPS_TAILSCALE_IP:/opt/polymarket/app/.env

# 5. Secure the .env file
chmod 600 /opt/polymarket/app/.env

# 6. First build and launch
docker compose up -d --build

# 7. Verify
docker compose ps
docker compose logs -f streak-bot
```

---

## Ongoing Dev → Deploy Workflow

```bash
# --- LOCAL machine ---
git push origin main

# --- VPS (SSH via Tailscale) ---
cd /opt/polymarket/app
git pull origin main
docker compose up -d --build

# Build time:
#   Only scripts/packages changed (no new deps): ~15 seconds
#   pyproject.toml / uv.lock changed (new dependency): ~2–5 minutes
```

---

## Operational Reference

```bash
# View logs
docker compose logs -f streak-bot           # live tail
docker compose logs streak-bot --tail=100   # last 100 lines
docker compose logs --since=1h              # all services, last hour

# Restart / stop
docker compose restart streak-bot
docker compose stop                         # graceful stop (SIGTERM → state saved)
docker compose down                         # remove containers, volumes preserved

# Enable a second bot
# 1. Uncomment its block in docker-compose.yml
docker compose up -d copy-bot               # starts only the new service

# Inspect trade state
cat /opt/polymarket/state/trades.json | python3 -m json.tool

# Update .env (e.g. change settings or go live) — no rebuild needed
scp .env VPS:/opt/polymarket/app/.env
docker compose up -d                        # recreates containers with new env

# Clean up old images after updates
docker image prune -f
```

---

## Verification After First Deployment

1. `docker compose ps` → shows `streak-bot` as `Up`
2. `docker compose logs streak-bot --tail=30` → bot started, no import errors
3. `ls /opt/polymarket/state/` → `trades.json` and `bot.log` appear within ~30 sec
4. `cat /opt/polymarket/state/trades.json` → confirms state file is in the mounted volume
5. `docker compose restart streak-bot && cat /opt/polymarket/state/trades.json` → state survived restart

---

## Architecture Notes

| Concern | Decision | Reason |
|---|---|---|
| uv install | `COPY --from=ghcr.io/astral-sh/uv` | Pinned, official, no pip overhead |
| Layer caching | Metadata stubs → `uv sync` → copy source | Avoids reinstalling 241 packages on every code edit |
| Secrets | `env_file: .env` in Compose, `.dockerignore` blocks it | Never baked into image |
| State persistence | Bind mounts at `/opt/polymarket/state` | Easy to inspect, back up, and `scp` locally |
| Multi-bot config | YAML anchors + commented service blocks | One-line enable/disable per bot |
| Restart policy | `unless-stopped` | Survives VPS reboots; stops cleanly on `docker compose stop` |
| Log rotation | `json-file` 10 MB / 5 files | 50 MB max per bot, no extra tooling |
