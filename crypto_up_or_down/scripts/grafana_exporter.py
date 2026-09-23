#!/usr/bin/env python3
# ruff: noqa: I001
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
  polymarket_state_read_success    — whether each strategy state file is readable

Labels: strategy, paper ("true"/"false")

Usage:
    python grafana_exporter.py [--state-dir DIR] [--port PORT]
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from collections import defaultdict
from http.server import HTTPServer
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from grafana_registry import retired_strategies  # noqa: E402

from prometheus_client import REGISTRY, MetricsHandler  # noqa: E402
from prometheus_client.core import GaugeMetricFamily  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STATE_DIR = os.environ.get("STATE_DIR", "/opt/polymarket/state")
PORT = int(os.environ.get("EXPORTER_PORT", "8002"))
OPEN_ORDER_STATUSES = {"pending", "submitted", "live", "open"}
UNRESOLVED_ORDER_STATUSES = {"unknown", "cancel_failed"}


def _as_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _latest_ledger_state(path: str) -> tuple[dict[str, dict], bool]:
    intents: dict[str, dict] = {}
    latest: dict[str, dict] = {}
    if not os.path.exists(path):
        return {}, True

    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("type") == "order_intent":
                    intent = record.get("intent") or {}
                    intent_id = intent.get("id")
                    if intent_id:
                        intents[intent_id] = intent
                        latest.setdefault(
                            intent_id,
                            {
                                "strategy": intent.get("strategy", "unknown"),
                                "status": "pending",
                                "amount_usd": _as_float(intent.get("amount_usd")),
                            },
                        )
                elif record.get("type") == "order_event":
                    intent_id = record.get("intent_id")
                    if not intent_id:
                        continue
                    intent = intents.get(intent_id, {})
                    latest[intent_id] = {
                        "strategy": intent.get("strategy", "unknown"),
                        "status": str(record.get("status") or "unknown").lower(),
                        "amount_usd": _as_float(intent.get("amount_usd")),
                    }
    except (OSError, json.JSONDecodeError) as e:
        log.warning("could not read order ledger %s: %s", path, e)
        return {}, False
    return latest, True


def _reconciliation_rows(state_dir: str) -> list[tuple[str, float, float]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(state_dir, "*reconciliation*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("could not read reconciliation state %s: %s", path, e)
            rows.append((os.path.basename(path), 0.0, 1.0))
            continue

        status = str(data.get("status", "")).lower()
        stale = data.get("stale")
        failed = data.get("failed")
        if stale is None:
            stale = status == "stale"
        if failed is None:
            failed = status in {"failed", "fail", "error"}
        rows.append((os.path.basename(path), float(bool(stale)), float(bool(failed))))
    return rows


def _health_rows(state_dir: str) -> list[tuple[str, float | None, float, bool]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(state_dir, "*-health.json"))):
        strategy = os.path.basename(path).replace("-health.json", "")
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("could not read health state %s: %s", path, e)
            rows.append((strategy, None, 0.0, False))
            continue

        quote_age = data.get("quote_age_sec")
        rows.append(
            (
                str(data.get("strategy") or strategy),
                _as_float(quote_age) if quote_age is not None else None,
                _as_float(data.get("api_errors_total")),
                True,
            )
        )
    return rows


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
        state_read_success_g = GaugeMetricFamily(
            "polymarket_state_read_success",
            "Whether the strategy state file is readable (1=yes, 0=no)",
            labels=["strategy"],
        )
        order_status_g = GaugeMetricFamily(
            "polymarket_order_ledger_status",
            "Latest order ledger count by strategy and status",
            labels=["strategy", "status"],
        )
        order_unknown_g = GaugeMetricFamily(
            "polymarket_order_unknown",
            "Order ledger entries with latest status unknown",
            labels=["strategy"],
        )
        open_orders_g = GaugeMetricFamily(
            "polymarket_open_orders",
            "Open or unresolved order count",
            labels=["strategy", "source"],
        )
        exposure_g = GaugeMetricFamily(
            "polymarket_exposure_usd",
            "Open or pending exposure in USD",
            labels=["strategy", "source"],
        )
        cancel_failures_g = GaugeMetricFamily(
            "polymarket_cancel_failures",
            "Cancel failures observed in state or order ledger",
            labels=["strategy", "source"],
        )
        kill_switch_g = GaugeMetricFamily(
            "polymarket_live_kill_switch_active",
            "Whether a live kill switch is active",
            labels=["source"],
        )
        reconciliation_stale_g = GaugeMetricFamily(
            "polymarket_reconciliation_stale",
            "Whether reconciliation state is stale when a state file exists",
            labels=["source"],
        )
        reconciliation_failed_g = GaugeMetricFamily(
            "polymarket_reconciliation_failed",
            "Whether reconciliation state is failed when a state file exists",
            labels=["source"],
        )
        ledger_read_success_g = GaugeMetricFamily(
            "polymarket_order_ledger_read_success",
            "Whether the order ledger is readable (1=yes, 0=no)",
            labels=[],
        )
        quote_age_g = GaugeMetricFamily(
            "polymarket_quote_age_sec",
            "Latest quote fetch duration or age in seconds",
            labels=["strategy"],
        )
        api_errors_g = GaugeMetricFamily(
            "polymarket_api_errors_total",
            "API errors reported by strategy health files",
            labels=["strategy"],
        )
        health_read_success_g = GaugeMetricFamily(
            "polymarket_health_read_success",
            "Whether strategy health files are readable (1=yes, 0=no)",
            labels=["strategy"],
        )

        retired = retired_strategies()
        pattern = os.path.join(self.state_dir, "*-trades.json")
        state_cancel_failures: defaultdict[str, float] = defaultdict(float)
        state_exposure: defaultdict[tuple[str, str], float] = defaultdict(float)
        for path in sorted(glob.glob(pattern)):
            strategy = os.path.basename(path).replace("-trades.json", "")
            if strategy in retired:
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                log.warning("could not read %s: %s", path, e)
                state_read_success_g.add_metric([strategy], 0)
                continue

            if not isinstance(data, dict):
                state_read_success_g.add_metric([strategy], 0)
                continue
            state_read_success_g.add_metric([strategy], 1)

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
            for trade in trades:
                if trade.get("order_status") == "cancel_failed":
                    state_cancel_failures[strategy] += 1
                if trade.get("settlement", {}).get("status") == "pending":
                    state_exposure[(strategy, paper)] += _as_float(trade.get("position", {}).get("amount"))

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

        ledger_path = os.environ.get("ORDER_LEDGER_FILE") or os.path.join(self.state_dir, "order_ledger.jsonl")
        latest_orders, ledger_ok = _latest_ledger_state(ledger_path)
        ledger_read_success_g.add_metric([], 1.0 if ledger_ok else 0.0)

        ledger_status_counts: defaultdict[tuple[str, str], float] = defaultdict(float)
        ledger_open_counts: defaultdict[str, float] = defaultdict(float)
        ledger_open_exposure: defaultdict[str, float] = defaultdict(float)
        ledger_cancel_failures: defaultdict[str, float] = defaultdict(float)
        ledger_unknowns: defaultdict[str, float] = defaultdict(float)
        for order in latest_orders.values():
            strategy = str(order.get("strategy") or "unknown")
            status = str(order.get("status") or "unknown").lower()
            amount = _as_float(order.get("amount_usd"))
            ledger_status_counts[(strategy, status)] += 1
            if status == "unknown":
                ledger_unknowns[strategy] += 1
            if status in OPEN_ORDER_STATUSES or status in UNRESOLVED_ORDER_STATUSES:
                ledger_open_counts[strategy] += 1
                ledger_open_exposure[strategy] += amount
            if status == "cancel_failed":
                ledger_cancel_failures[strategy] += 1

        for (strategy, status), count in sorted(ledger_status_counts.items()):
            order_status_g.add_metric([strategy, status], count)
        for strategy, count in sorted(ledger_unknowns.items()):
            order_unknown_g.add_metric([strategy], count)
        for strategy, count in sorted(ledger_open_counts.items()):
            open_orders_g.add_metric([strategy, "ledger"], count)
        for strategy, exposure in sorted(ledger_open_exposure.items()):
            exposure_g.add_metric([strategy, "ledger"], exposure)
        for strategy, count in sorted(ledger_cancel_failures.items()):
            cancel_failures_g.add_metric([strategy, "ledger"], count)
        for strategy, count in sorted(state_cancel_failures.items()):
            cancel_failures_g.add_metric([strategy, "state"], count)
        for (strategy, paper), exposure in sorted(state_exposure.items()):
            exposure_g.add_metric([strategy, f"state:paper={paper}"], exposure)

        kill_switch_g.add_metric(
            ["env"],
            1.0 if os.environ.get("LIVE_KILL_SWITCH", "").lower() in {"1", "true", "yes", "on"} else 0.0,
        )
        kill_file = os.environ.get("LIVE_KILL_SWITCH_FILE", "").strip()
        kill_switch_g.add_metric(["file"], 1.0 if kill_file and os.path.exists(kill_file) else 0.0)

        for source, stale, failed in _reconciliation_rows(self.state_dir):
            reconciliation_stale_g.add_metric([source], stale)
            reconciliation_failed_g.add_metric([source], failed)

        for strategy, quote_age, api_errors, health_ok in _health_rows(self.state_dir):
            health_read_success_g.add_metric([strategy], 1.0 if health_ok else 0.0)
            api_errors_g.add_metric([strategy], api_errors)
            if quote_age is not None:
                quote_age_g.add_metric([strategy], quote_age)

        yield bankroll_g
        yield daily_pnl_g
        yield daily_bets_g
        yield total_trades_g
        yield total_pnl_g
        yield win_rate_g
        yield avg_stake_g
        yield last_trade_age_g
        yield consecutive_losses_g
        yield state_read_success_g
        yield order_status_g
        yield order_unknown_g
        yield open_orders_g
        yield exposure_g
        yield cancel_failures_g
        yield kill_switch_g
        yield reconciliation_stale_g
        yield reconciliation_failed_g
        yield ledger_read_success_g
        yield quote_age_g
        yield api_errors_g
        yield health_read_success_g


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
