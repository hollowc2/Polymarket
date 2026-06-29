"""Append-only order intent ledger for live execution."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OrderIntent:
    """Pre-submit live order intent."""

    id: str
    strategy: str
    market_slug: str
    token_id: str
    direction: str
    side: str
    amount_usd: float
    max_price: float
    created_at_ms: int


@dataclass(frozen=True)
class OrderLedgerEvent:
    """Single append-only order lifecycle event."""

    event: str
    intent_id: str
    timestamp_ms: int
    status: str
    order_id: str | None = None
    reason: str = ""
    payload: dict[str, Any] | None = None


class JsonOrderLedger:
    """Durable JSONL ledger.

    Each write is one complete JSON object plus fsync. This keeps the first
    version simple while still giving startup reconciliation a durable event log.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def record_intent(self, intent: OrderIntent) -> None:
        self._append(
            {
                "type": "order_intent",
                "intent": asdict(intent),
                "timestamp_ms": _now_ms(),
            }
        )

    def record_event(self, event: OrderLedgerEvent) -> None:
        self._append(
            {
                "type": "order_event",
                **asdict(event),
            }
        )

    def _append(self, record: dict[str, Any]) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def _now_ms() -> int:
    return int(time.time() * 1000)
