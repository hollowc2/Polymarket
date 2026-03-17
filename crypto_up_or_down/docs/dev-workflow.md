# Dev Workflow — Polymarket Auto Trader

## Overview

Code locally on your desktop, push to GitHub, pull and deploy on the VPS.
The VPS runs bots 24/7 in Docker containers.

```
Desktop (code + test)  →  git push  →  VPS (git pull + docker compose up -d --build)
```

---

## The Basic Dev Cycle

### 1. Write and test locally

```bash
# Run in paper mode locally to test your changes
uv run python scripts/bots/streak_bot.py --paper
```

### 2. Commit and push

```bash
git add packages/ scripts/
git commit -m "feat: your change"
PATH="$(uv run python -c 'import sys, os; print(os.path.dirname(sys.executable))'):$PATH" git push origin main
```

> If you're inside `nix develop`, just use `git push` directly — ruff and ty are on PATH automatically.

### 3. Deploy on VPS

```bash
ssh billy@YOUR_TAILSCALE_IP
cd /opt/polymarket/app
git pull
docker compose up -d --build
```

### 4. Verify

```bash
docker compose logs -f streak-bot
```

Hit `Ctrl+C` to stop watching — the bot keeps running.

**Build times:**
- Code-only change (`scripts/`, `packages/` source): ~15 seconds
- New dependency added (`pyproject.toml` / `uv.lock` changed): ~2–5 minutes

---

## Adding a New Bot

### 1. Write the bot script

Create `scripts/bots/my_bot.py`. Follow the same pattern as `scripts/bots/streak_bot.py`.

### 2. Add it to docker-compose.yml

Uncomment the template block at the bottom of `docker-compose.yml` and fill in your details:

```yaml
  my-new-bot:
    <<: *bot-common
    container_name: polymarket-my-new-bot
    command: ["uv", "run", "python", "scripts/bots/my_bot.py", "--paper"]
    environment:
      PAPER_TRADE: "true"
      TRADES_FILE: "/app/state/my-bot-trades.json"
      HISTORY_FILE: "/app/state/my-bot-history.json"
      LOG_FILE: "/app/state/my-bot.log"
```

Each bot gets its own state files so they never collide with each other.

### 3. Commit and push

```bash
git add scripts/my_bot.py docker-compose.yml
git commit -m "feat: add my new bot"
PATH="$(uv run python -c 'import sys, os; print(os.path.dirname(sys.executable))'):$PATH" git push origin main
```

### 4. Deploy on VPS

```bash
cd /opt/polymarket/app
git pull
docker compose up -d --build
```

Docker builds the updated image and starts the new container.
All other running bots are unaffected.

### 5. Verify both are running

```bash
docker compose ps
```

```
NAME                       STATUS
polymarket-streak-bot      Up 2 hours
polymarket-my-new-bot      Up 10 seconds
```

---

## Stopping and Disabling Bots

### Stop temporarily (easy to restart)

```bash
docker compose stop my-new-bot
```

Restart it later with:

```bash
docker compose start my-new-bot
```

The container still exists and will come back exactly where it left off.

### Disable permanently (comment out in compose)

In `docker-compose.yml`, comment out the entire service block:

```yaml
  # my-new-bot:
  #   <<: *bot-common
  #   container_name: polymarket-my-new-bot
  #   ...
```

Then on the VPS:

```bash
git pull   # if you edited docker-compose.yml locally and pushed
docker compose down my-new-bot
docker compose up -d
```

**Which to use:**
| Situation | Method |
|---|---|
| Pausing a bot for a few hours/days | `docker compose stop` |
| Bot is broken, fixing it | `docker compose stop` |
| Done with this bot indefinitely | Comment out in compose |
| Temporarily cutting costs/resources | `docker compose stop` |

Either way, state files in `/opt/polymarket/state/` are untouched —
trade history and bankroll are safe regardless of what you do to the container.

---

## Checking What's Running

```bash
# Status of all bots
docker compose ps

# Live logs for a specific bot
docker compose logs -f streak-bot

# Resource usage (CPU, memory per container)
docker stats
```

---

## Quick Reference

| Task | Command |
|---|---|
| Deploy update | `git pull && docker compose up -d --build` |
| View logs | `docker compose logs -f <bot-name>` |
| Restart a bot | `docker compose restart <bot-name>` |
| Stop a bot | `docker compose stop <bot-name>` |
| Start a stopped bot | `docker compose start <bot-name>` |
| Stop everything | `docker compose stop` |
| Check status | `docker compose ps` |
| Inspect trade state | `cat /opt/polymarket/state/trades.json \| python3 -m json.tool` |
