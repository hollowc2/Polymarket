# ruff: noqa: I001
import json
import sys
from collections import namedtuple
from types import ModuleType


Sample = namedtuple("Sample", "name labels value")


class GaugeMetricFamily:
    def __init__(self, name, _description, labels):
        self.name = name
        self._labels = labels
        self.samples = []

    def add_metric(self, values, value):
        self.samples.append(
            Sample(self.name, dict(zip(self._labels, values, strict=False)), value)
        )


prometheus_client = ModuleType("prometheus_client")
prometheus_client.REGISTRY = type("Registry", (), {"register": lambda *_: None})()
prometheus_client.MetricsHandler = object
prometheus_core = ModuleType("prometheus_client.core")
prometheus_core.GaugeMetricFamily = GaugeMetricFamily
sys.modules.setdefault("prometheus_client", prometheus_client)
sys.modules.setdefault("prometheus_client.core", prometheus_core)

from scripts.grafana_exporter import PolymarketCollector  # noqa: E402


def _metric_samples(tmp_path, metric_name):
    return {
        tuple(sorted(sample.labels.items())): sample.value
        for metric in PolymarketCollector(str(tmp_path)).collect()
        if metric.name == metric_name
        for sample in metric.samples
    }


def _state_read_samples(tmp_path):
    return {
        dict(labels)["strategy"]: value
        for labels, value in _metric_samples(tmp_path, "polymarket_state_read_success").items()
    }


def test_state_read_success_reports_valid_and_invalid_files(tmp_path):
    (tmp_path / "valid-strategy-trades.json").write_text(json.dumps({"trades": []}))
    (tmp_path / "broken-strategy-trades.json").write_text("{")

    assert _state_read_samples(tmp_path) == {
        "broken-strategy": 0,
        "valid-strategy": 1,
    }


def test_phase8_observability_metrics_from_state_and_ledger(tmp_path, monkeypatch):
    kill_file = tmp_path / "kill"
    kill_file.write_text("stop", encoding="utf-8")
    monkeypatch.setenv("LIVE_KILL_SWITCH", "true")
    monkeypatch.setenv("LIVE_KILL_SWITCH_FILE", str(kill_file))
    monkeypatch.setenv("ORDER_LEDGER_FILE", str(tmp_path / "order_ledger.jsonl"))

    (tmp_path / "live-strategy-trades.json").write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "context": {"mode": "live"},
                        "order_status": "cancel_failed",
                        "position": {"amount": 7.5},
                        "settlement": {"status": "pending"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "order_ledger.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "order_intent",
                        "intent": {
                            "id": "live-strategy:market:up:1",
                            "strategy": "live-strategy",
                            "amount_usd": 5.0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "order_event",
                        "intent_id": "live-strategy:market:up:1",
                        "status": "unknown",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "startup-reconciliation.json").write_text(
        json.dumps({"status": "failed", "stale": True}),
        encoding="utf-8",
    )

    assert _metric_samples(tmp_path, "polymarket_order_ledger_status") == {
        (("status", "unknown"), ("strategy", "live-strategy")): 1
    }
    assert _metric_samples(tmp_path, "polymarket_order_unknown") == {
        (("strategy", "live-strategy"),): 1
    }
    assert _metric_samples(tmp_path, "polymarket_open_orders") == {
        (("source", "ledger"), ("strategy", "live-strategy")): 1
    }
    assert _metric_samples(tmp_path, "polymarket_exposure_usd") == {
        (("source", "ledger"), ("strategy", "live-strategy")): 5.0,
        (("source", "state:paper=false"), ("strategy", "live-strategy")): 7.5,
    }
    assert _metric_samples(tmp_path, "polymarket_cancel_failures") == {
        (("source", "state"), ("strategy", "live-strategy")): 1
    }
    assert _metric_samples(tmp_path, "polymarket_live_kill_switch_active") == {
        (("source", "env"),): 1.0,
        (("source", "file"),): 1.0,
    }
    assert _metric_samples(tmp_path, "polymarket_reconciliation_stale") == {
        (("source", "startup-reconciliation.json"),): 1.0
    }
    assert _metric_samples(tmp_path, "polymarket_reconciliation_failed") == {
        (("source", "startup-reconciliation.json"),): 1.0
    }


def test_phase8_health_metrics_from_strategy_health_file(tmp_path):
    (tmp_path / "impulse-momentum-health.json").write_text(
        json.dumps(
            {
                "strategy": "impulse-momentum",
                "quote_age_sec": 0.321,
                "api_errors_total": 2,
            }
        ),
        encoding="utf-8",
    )

    assert _metric_samples(tmp_path, "polymarket_quote_age_sec") == {
        (("strategy", "impulse-momentum"),): 0.321
    }
    assert _metric_samples(tmp_path, "polymarket_api_errors_total") == {
        (("strategy", "impulse-momentum"),): 2.0
    }
    assert _metric_samples(tmp_path, "polymarket_health_read_success") == {
        (("strategy", "impulse-momentum"),): 1.0
    }
