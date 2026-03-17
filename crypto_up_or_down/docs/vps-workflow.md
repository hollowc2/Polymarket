# VPS Workflow — Polymarket Auto Trader

## Overview

Bots run in Docker containers on a Hostinger VPS (Ubuntu 24.04).
Access is via Tailscale SSH. Development happens locally — changes are pushed
to GitHub and pulled onto the VPS manually.

```
Local machine  →  git push  →  GitHub  →  git pull (VPS)  →  docker compose up -d --build
```

---

## VPS Connection

```bash
ssh billy@YOUR_TAILSCALE_IP
```

Find your Tailscale IP anytime with:

```bash
tailscale ip -4
```

---

## Deploy an Update

Run these on the **VPS** after pushing changes from your local machine:

```bash
cd /opt/polymarket/app
git pull
docker compose up -d --build
```

- If only `scripts/` or `packages/` source changed: rebuilds in ~15 seconds (deps are cached)
- If `pyproject.toml` or `uv.lock` changed (new dependency added): rebuilds in ~2–5 minutes

---

## View Logs

```bash
# Live tail (Ctrl+C to exit)
docker compose logs -f streak-bot

# Last 100 lines
docker compose logs streak-bot --tail=100

# All services, last hour
docker compose logs --since=1h
```

---

## Check Status

```bash
docker compose ps
```

Expected output when healthy:

```
NAME                      STATUS
polymarket-streak-bot     Up X hours
```

---

## Restart / Stop / Start

```bash
# Graceful restart (bot saves state before stopping)
docker compose restart streak-bot

# Stop all bots (graceful shutdown)
docker compose stop

# Start all bots (after a stop)
docker compose up -d

# Stop and remove containers (volumes and state are preserved)
docker compose down
```

---

## Inspect Trade State

State files live on the VPS host at `/opt/polymarket/state/` and persist
across container restarts and rebuilds.

```bash
# View current trade state (bankroll, open positions, daily stats)
cat /opt/polymarket/state/trades.json | python3 -m json.tool

# View full trade history
cat /opt/polymarket/state/trade_history_full.json | python3 -m json.tool

# View bot log file
tail -50 /opt/polymarket/state/bot.log

# Watch log file live
tail -f /opt/polymarket/state/bot.log
```

---

## Update .env Settings

No rebuild needed — just update the file and recreate the container:

```bash
# Edit directly on VPS
nano /opt/polymarket/app/.env

# Or copy updated .env from local machine
scp .env billy@YOUR_TAILSCALE_IP:/opt/polymarket/app/.env

# Restart containers to pick up new env
docker compose up -d
```

---

## Go Live (Disable Paper Trading)

When you're ready to trade with real money:

1. Add your `PRIVATE_KEY` to `/opt/polymarket/app/.env`
2. In `docker-compose.yml`, remove or comment out the `PAPER_TRADE: "true"` line
   under the bot's `environment:` block (the line in `.env` is `PAPER_TRADE=true`
   by default — you need to remove the override in compose too)
3. Restart the bot:

```bash
docker compose up -d
```

> **Warning:** Double-check `docker compose logs streak-bot --tail=5` after restarting
> to confirm it shows `Live trading mode` before walking away.

---

## Add a New Bot

1. Write your bot script at `scripts/my_bot.py`
2. Open `docker-compose.yml` and uncomment the **New Bot Template** block,
   renaming it and updating the command
3. Push and deploy:

```bash
# Local
git push origin main

# VPS
cd /opt/polymarket/app
git pull
docker compose up -d --build
```

Only the new bot's container is created — running bots are not affected.

---

## Backup State Files

Before any risky operation (rebuilding from scratch, changing .env significantly):

```bash
cp /opt/polymarket/state/trades.json \
   /opt/polymarket/state/trades.json.bak.$(date +%Y%m%d)

cp /opt/polymarket/state/trade_history_full.json \
   /opt/polymarket/state/trade_history_full.json.bak.$(date +%Y%m%d)
```

---

## Disk & Resource Usage

```bash
# Overall disk usage for polymarket data
du -sh /opt/polymarket/

# Docker image sizes
docker images | grep polymarket

# Container resource usage (CPU, memory)
docker stats

# Clean up old/untagged images after updates (safe)
docker image prune -f
```

---

## Full Rebuild from Scratch

Only needed if the image is corrupted or you want a completely clean slate.
State files are preserved on the host — they are not inside the container.

```bash
docker compose down
docker compose up -d --build
```

---

## Directory Structure on VPS

```
/opt/polymarket/
├── app/                   ← git repo (cloned from GitHub)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── .env               ← secrets, never committed to git
│   ├── packages/
│   └── scripts/
├── state/                 ← persistent bot state (survives rebuilds)
│   ├── trades.json
│   ├── trade_history_full.json
│   └── bot.log
└── data/                  ← Binance OHLCV parquet files (survives rebuilds)
```

---

## Troubleshooting

**Bot container exits immediately:**
```bash
docker compose logs streak-bot --tail=50
# Look for Python import errors or missing env vars
```

**Permission denied on docker socket:**
```bash
sudo usermod -aG docker billy && newgrp docker
```

**State file is empty after restart:**
Check that the volume mount is correct — state files should be at
`/opt/polymarket/state/`, not inside the container:
```bash
ls -la /opt/polymarket/state/
```

**Bot not picking up .env changes:**
```bash
# 'docker compose up -d' (without --build) recreates containers with fresh env
docker compose up -d
```

**Out of disk space:**
```bash
docker image prune -f       # remove unused images
docker system prune -f      # remove unused images, networks, build cache
```
