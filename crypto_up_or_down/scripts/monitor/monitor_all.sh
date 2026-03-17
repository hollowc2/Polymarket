#!/usr/bin/env bash
# monitor_all.sh — one tmux window per active bot, 70/30 split (stats | logs)
# Usage: bash scripts/monitor_all.sh          # launch / re-attach
#        bash scripts/monitor_all.sh --kill   # kill session

set -euo pipefail
export TERM=xterm-256color

SESSION="poly-monitors"

if [[ "${1:-}" == "--kill" ]]; then
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "Session killed." || echo "No active session."
  exit 0
fi

# Auto-detect running polymarket containers
CONTAINERS=$(docker ps --format '{{.Names}}' | grep '^polymarket-' | grep -v 'polymarket-gluetun' | sort || true)
if [[ -z "$CONTAINERS" ]]; then
  echo "No running polymarket-* containers found. Start bots first."
  exit 1
fi

# If session already exists, just re-attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Attaching to existing session '$SESSION'..."
  tmux attach-session -t "$SESSION"
  exit 0
fi

WIN_IDX=0
FIRST=true

while IFS= read -r CONTAINER; do
  WIN_NAME="${CONTAINER#polymarket-}"   # e.g. "streak-bot"

  if $FIRST; then
    # First window: created with the session.
    # -x/-y required when detached (no PTY → no inherited size → "size missing" on split-window).
    # Hardcoded large defaults; tmux resizes to real terminal on attach.
    tmux new-session -d -s "$SESSION" -n "$WIN_NAME" -x 220 -y 50
    FIRST=false
  else
    tmux new-window -t "$SESSION" -n "$WIN_NAME"
  fi

  # Split: bottom pane = 30%, top pane stays at 70%
  # Use -l 30% not -p 30 — -p was removed in tmux 3.2 (Ubuntu 22.04+)
  tmux split-window -v -l 30% -t "$SESSION:$WIN_IDX"

  # Choose the right monitor script based on container name
  case "$CONTAINER" in
    *turtlequant*)
      MONITOR_CMD="uv run --script scripts/monitor/monitor_turtlequant.py --live --state-dir /app/state/turtlequant"
      ;;
    *slowquant*)
      MONITOR_CMD="uv run --script scripts/monitor/monitor_slowquant.py --live --state-dir /app/state/slowquant"
      ;;
    *)
      MONITOR_CMD="uv run --script scripts/monitor/monitor.py --live"
      ;;
  esac

  # Top pane (0) → live dashboard
  tmux send-keys -t "$SESSION:$WIN_IDX.0" \
    "docker exec -it $CONTAINER $MONITOR_CMD" Enter

  # Bottom pane (1) → docker log stream
  tmux send-keys -t "$SESSION:$WIN_IDX.1" \
    "docker logs -f $CONTAINER" Enter

  WIN_IDX=$((WIN_IDX + 1))
done <<< "$CONTAINERS"

# Start on first window
tmux select-window -t "$SESSION:0"
echo "Attaching to '$SESSION' — ${WIN_IDX} bot(s). Use Ctrl+B n/p or 0-$((WIN_IDX-1)) to switch tabs."
tmux attach-session -t "$SESSION"
