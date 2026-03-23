#!/usr/bin/env python3
"""Prometheus exporter for polymarket crypto_up_or_down live state.

Reads all *-trades.json files from STATE_DIR and exposes per-strategy gauges:
  polymarket_bankroll         — current bankroll USD
  polymarket_daily_pnl_usd    — today's net P&L USD
  polymarket_daily_bets       — number of bets placed today

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

        yield bankroll_g
        yield daily_pnl_g
        yield daily_bets_g


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
