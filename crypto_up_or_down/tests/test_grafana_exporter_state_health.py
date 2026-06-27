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


def _samples(tmp_path):
    return {
        sample.labels["strategy"]: sample.value
        for metric in PolymarketCollector(str(tmp_path)).collect()
        if metric.name == "polymarket_state_read_success"
        for sample in metric.samples
    }


def test_state_read_success_reports_valid_and_invalid_files(tmp_path):
    (tmp_path / "valid-strategy-trades.json").write_text(json.dumps({"trades": []}))
    (tmp_path / "broken-strategy-trades.json").write_text("{")

    assert _samples(tmp_path) == {
        "broken-strategy": 0,
        "valid-strategy": 1,
    }
