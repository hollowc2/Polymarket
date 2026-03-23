#!/usr/bin/env python3
"""Prometheus exporter for polymarket crypto_up_or_down live state.

Reads all *-trades.json files from STATE_DIR and exposes per-strategy gauges:
  polymarket_bankroll              — current bankroll USD
  polymarket_daily_pnl_usd         — today's net P&L USD
  polymarket_daily_bets            — number of bets placed today
  polymarket_total_trades          — all-time trade count
  polymarket_total_pnl_usd         — all-time net P&L USD
  polymarket_win_rate              — fraction of settled trades won (0–1)
  polymarket_avg_stake_usd         — mean stake per trade USD
  polymarket_last_trade_age_sec    — seconds since last trade execution
  polymarket_consecutive_losses    — consecutive losses at last trade

Labels: strategy, paper ("true"/"false")

Usage:
    python grafana_exporter.py [--state-dir DIR] [--port PORT]
"""

import argparse
import glob
import json
import logging
import os
import time
from http.server import HTTPServer

from prometheus_client import REGISTRY, MetricsHandler
from prometheus_client.core import GaugeMetricFamily

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STATE_DIR = os.environ.get("STATE_DIR", "/opt/polymarket/state")
PORT = int(os.environ.get("EXPORTER_PORT", "8002"))


class PolymarketCollector:
    def __init__(self, state_dir: str):
        self.state_dir = state_dir

    def collect(self):
        bankroll_g = GaugeMetricFamily(
            "polymarket_bankroll",
            "Current bankroll in USD",
            labels=["strategy", "paper"],
        )
        daily_pnl_g = GaugeMetricFamily(
            "polymarket_daily_pnl_usd",
            "Today's net P&L in USD",
            labels=["strategy", "paper"],
        )
        daily_bets_g = GaugeMetricFamily(
            "polymarket_daily_bets",
            "Number of bets placed today",
            labels=["strategy", "paper"],
        )
        total_trades_g = GaugeMetricFamily(
            "polymarket_total_trades",
            "All-time trade count",
            labels=["strategy", "paper"],
        )
        total_pnl_g = GaugeMetricFamily(
            "polymarket_total_pnl_usd",
            "All-time net P&L in USD",
            labels=["strategy", "paper"],
        )
        win_rate_g = GaugeMetricFamily(
            "polymarket_win_rate",
            "Fraction of settled trades won (0-1)",
            labels=["strategy", "paper"],
        )
        avg_stake_g = GaugeMetricFamily(
            "polymarket_avg_stake_usd",
            "Mean stake per trade in USD",
            labels=["strategy", "paper"],
        )
        last_trade_age_g = GaugeMetricFamily(
            "polymarket_last_trade_age_sec",
            "Seconds since last trade execution",
            labels=["strategy", "paper"],
        )
        consecutive_losses_g = GaugeMetricFamily(
            "polymarket_consecutive_losses",
            "Consecutive losses at time of last trade",
            labels=["strategy", "paper"],
        )

        pattern = os.path.join(self.state_dir, "*-trades.json")
        for path in sorted(glob.glob(pattern)):
            strategy = os.path.basename(path).replace("-trades.json", "")
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                log.warning("could not read %s: %s", path, e)
                continue

            if not isinstance(data, dict):
                continue

            # Determine paper/live from the most recent trade in the list
            trades = data.get("trades") or []
            paper = "true"
            if trades:
                latest = trades[-1]
                ctx = latest.get("context") or {}
                paper = "false" if ctx.get("mode") == "live" else "true"

            labels = [strategy, paper]
            bankroll = data.get("bankroll")
            daily_pnl = data.get("daily_pnl")
            daily_bets = data.get("daily_bets")

            if bankroll is not None:
                bankroll_g.add_metric(labels, float(bankroll))
            if daily_pnl is not None:
                daily_pnl_g.add_metric(labels, float(daily_pnl))
            if daily_bets is not None:
                daily_bets_g.add_metric(labels, float(daily_bets))

            total_trades_g.add_metric(labels, len(trades))

            settled = [t for t in trades if t.get("settlement", {}).get("status") == "settled"]
            if settled:
                wins = sum(1 for t in settled if t.get("settlement", {}).get("won"))
                total_pnl = sum(t.get("settlement", {}).get("net_profit", 0.0) for t in settled)
                avg_stake = sum(t.get("position", {}).get("amount", 0.0) for t in settled) / len(settled)
                total_pnl_g.add_metric(labels, float(total_pnl))
                win_rate_g.add_metric(labels, wins / len(settled))
                avg_stake_g.add_metric(labels, float(avg_stake))

            if trades:
                last_exec_ts_ms = trades[-1].get("execution", {}).get("timestamp")
                if last_exec_ts_ms is not None:
                    last_trade_age_g.add_metric(labels, time.time() - last_exec_ts_ms / 1000.0)
                consec_losses = trades[-1].get("session", {}).get("consecutive_losses")
                if consec_losses is not None:
                    consecutive_losses_g.add_metric(labels, float(consec_losses))

        yield bankroll_g
        yield daily_pnl_g
        yield daily_bets_g
        yield total_trades_g
        yield total_pnl_g
        yield win_rate_g
        yield avg_stake_g
        yield last_trade_age_g
        yield consecutive_losses_g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=STATE_DIR)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    REGISTRY.register(PolymarketCollector(args.state_dir))

    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    log.info("Polymarket Prometheus exporter listening on :%d", args.port)
    log.info("State dir: %s", args.state_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
