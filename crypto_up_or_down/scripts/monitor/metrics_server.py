#!/usr/bin/env python3
"""Metrics HTTP server for Grafana Infinity datasource.

Reads bot state files and exposes computed stats as JSON.

Endpoints:
  GET /health                  → {"status": "ok"}
  GET /api/leaderboard         → all bot stats (one object per bot)
  GET /api/trades?bot=<label>  → trades for one bot (newest-first, with cumulative P&L)
  GET /api/bots                → bot label list (for variable queries)

Usage:
  uv run python scripts/monitor/metrics_server.py
  uv run python scripts/monitor/metrics_server.py --state-dir /opt/polymarket/state --port 9099
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── Bot registry (mirrors live_stats_tui.py) ──────────────────────────────────

BOTS: list[dict[str, str]] = [
    {"label": "streak-bot", "file": "trade_history_full.json"},
    {"label": "adx-eth-5m", "file": "adx-eth-5m-history.json"},
    {"label": "hl-orderflow-momo-5m", "file": "hl-orderflow-momo-5m-history.json"},
    {"label": "hl-orderflow-momo-15m", "file": "hl-orderflow-momo-15m-history.json"},
    {"label": "hl-orderflow-reversal-5m", "file": "hl-orderflow-reversal-5m-history.json"},
    {"label": "3barmomo-hl-5m", "file": "3barmomo-hl-5m-history.json"},
    {"label": "pinbar-hl-5m", "file": "pinbar-hl-5m-history.json"},
    {"label": "delta-flip-btc-5m", "file": "delta-flip-btc-5m-history.json"},
]

DEFAULT_STATE_DIR = "/opt/polymarket/state"
DEFAULT_PORT = 9099

POLYMARKET_FEE_RATE_BPS = 200


# ── Data loading ──────────────────────────────────────────────────────────────


def load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def parse_settled(raw: list[dict]) -> list[dict]:
    """Extract settled trades as plain dicts."""
    out: list[dict] = []
    for item in raw:
        try:
            exec_ = item.get("execution", {})
            settlement = item.get("settlement", {})
            market = item.get("market", {})
            gate = item.get("gate") or {}
            position = item.get("position", {})

            if settlement.get("status") not in ("settled", "forced_exit"):
                continue
            if settlement.get("won") is None:
                continue

            ts = int(exec_.get("timestamp", 0) or market.get("timestamp", 0))
            out.append(
                {
                    "fill": float(exec_.get("fill_price", exec_.get("entry_price", 0.5))),
                    "won": bool(settlement.get("won", False)),
                    "net_pnl": float(settlement.get("net_profit", 0.0)),
                    "ts": ts,
                    "market_slug": str(market.get("slug", item.get("market", ""))),
                    "direction": str(position.get("direction", "")),
                    "amount": float(position.get("amount", 0.0)),
                    "gate_name": gate.get("name", ""),
                    "gate_boosted": gate.get("boosted"),
                }
            )
        except Exception:
            continue
    return out


# ── Stats ─────────────────────────────────────────────────────────────────────


def _wilson_margin(wins: int, n: int) -> float:
    if n == 0:
        return 0.0
    z = 1.96
    return z * math.sqrt(wins * (n - wins) / n + z**2 / 4) / (n + z**2)


def _breakeven_wr(avg_fill: float) -> float:
    if avg_fill <= 0:
        return 1.0
    fee_pct = (POLYMARKET_FEE_RATE_BPS / 10000) * avg_fill * (1 - avg_fill)
    return avg_fill / ((1 - avg_fill) * (1 - fee_pct) + avg_fill)


def compute_stats(label: str, trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "bot": label,
            "n": 0,
            "win_rate": None,
            "ci_margin": None,
            "be_wr": None,
            "edge": None,
            "total_pnl": 0.0,
            "avg_pnl": None,
            "sharpe": None,
            "max_drawdown": None,
            "max_consec_losses": None,
        }

    wins = sum(1 for t in trades if t["won"])
    win_rate = wins / n

    fills = [t["fill"] for t in trades if t["fill"] > 0]
    avg_fill = sum(fills) / len(fills) if fills else 0.5
    be_wr = _breakeven_wr(avg_fill)
    edge = win_rate - be_wr

    pnls = [t["net_pnl"] for t in trades]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / n

    mean = avg_pnl
    variance = sum((p - mean) ** 2 for p in pnls) / max(n - 1, 1)
    std = math.sqrt(variance)
    sharpe = (mean / std * math.sqrt(n)) if std > 0 else 0.0

    peak = cum = max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    sorted_ts = sorted(trades, key=lambda t: t["ts"])
    mx = cur = 0
    for t in sorted_ts:
        if not t["won"]:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0

    return {
        "bot": label,
        "n": n,
        "win_rate": round(win_rate, 4),
        "ci_margin": round(_wilson_margin(wins, n), 4),
        "be_wr": round(be_wr, 4),
        "edge": round(edge, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "max_consec_losses": mx,
    }


# ── Trade list ────────────────────────────────────────────────────────────────


def build_trade_list(trades: list[dict]) -> list[dict]:
    """Return trades newest-first with running cumulative P&L computed chronologically."""
    chrono = sorted(trades, key=lambda t: t["ts"])

    # Compute cumulative P&L in chronological order
    cum_by_id: dict[int, float] = {}
    cum = 0.0
    for i, t in enumerate(chrono):
        cum += t["net_pnl"]
        cum_by_id[id(t)] = round(cum, 2)

    result: list[dict] = []
    for row_num, t in enumerate(reversed(chrono), 1):
        ts = t["ts"]
        dt = datetime.fromtimestamp(ts / 1000, tz=UTC).strftime("%m-%d %H:%M") if ts > 0 else "—"

        gate_str = t.get("gate_name", "") or "—"
        boosted = t.get("gate_boosted")
        if boosted is True:
            gate_str += " ✓"
        elif boosted is False and gate_str != "—":
            gate_str += " ✗"

        result.append(
            {
                "row": row_num,
                "datetime": dt,
                "wl": "W" if t["won"] else "L",
                "fill": round(t["fill"], 3),
                "net_pnl": round(t["net_pnl"], 2),
                "cum_pnl": cum_by_id[id(t)],
                "size": round(t["amount"], 2),
                "direction": t["direction"].upper() if t["direction"] else "—",
                "market_slug": t["market_slug"],
                "gate": gate_str,
            }
        )
    return result


# ── HTTP handler ──────────────────────────────────────────────────────────────


class MetricsHandler(BaseHTTPRequestHandler):
    state_dir: Path
    bot_map: dict[str, str]  # label → filename

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # suppress per-request access logs

    def _json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path

        if path == "/health":
            self._json({"status": "ok", "ts": datetime.now(UTC).isoformat()})

        elif path == "/api/bots":
            self._json([b["label"] for b in BOTS])

        elif path == "/api/leaderboard":
            rows = []
            for bot in BOTS:
                raw = load_raw(self.state_dir / bot["file"])
                trades = parse_settled(raw)
                rows.append(compute_stats(bot["label"], trades))
            self._json(rows)

        elif path == "/api/trades":
            bot_label = (params.get("bot") or [None])[0]
            if not bot_label:
                self._json({"error": "?bot= required"}, 400)
                return
            file_name = self.bot_map.get(bot_label)
            if not file_name:
                self._json({"error": f"unknown bot: {bot_label}"}, 404)
                return
            raw = load_raw(self.state_dir / file_name)
            trades = parse_settled(raw)
            self._json(build_trade_list(trades))

        else:
            self._json({"error": "not found"}, 404)


def _make_handler(state_dir: Path, bot_map: dict[str, str]) -> type[MetricsHandler]:
    class BoundHandler(MetricsHandler):
        pass

    BoundHandler.state_dir = state_dir
    BoundHandler.bot_map = bot_map
    return BoundHandler


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Metrics HTTP server for Grafana Infinity datasource")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR, metavar="DIR")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    bot_map = {b["label"]: b["file"] for b in BOTS}
    handler = _make_handler(state_dir, bot_map)

    print(f"Metrics server listening on {args.host}:{args.port}")
    print(f"State dir: {state_dir}")
    HTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
