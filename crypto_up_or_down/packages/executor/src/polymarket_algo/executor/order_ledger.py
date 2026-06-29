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

    def has_intent(self, intent_id: str) -> bool:
        return any(
            record.get("type") == "order_intent" and record.get("intent", {}).get("id") == intent_id
            for record in self._records()
        )

    def risk_snapshot(self) -> dict[str, Any]:
        intents: dict[str, dict[str, Any]] = {}
        status_by_intent: dict[str, str] = {}

        for record in self._records():
            if record.get("type") == "order_intent":
                intent = record.get("intent", {})
                if isinstance(intent, dict) and intent.get("id"):
                    intents[intent["id"]] = intent
            elif record.get("type") == "order_event":
                intent_id = record.get("intent_id")
                if intent_id:
                    status_by_intent[str(intent_id)] = str(record.get("status", ""))

        terminal = {"filled", "cancelled", "rejected", "failed"}
        open_statuses = {"", "pending", "submitted", "unknown", "cancel_failed"}
        market_exposure: dict[str, float] = {}
        strategy_exposure: dict[str, float] = {}
        total_notional = 0.0
        open_orders = 0

        for intent_id, intent in intents.items():
            status = status_by_intent.get(intent_id, "")
            if status in terminal:
                continue
            amount = float(intent.get("amount_usd", 0.0))
            market = str(intent.get("market_slug", ""))
            strategy = str(intent.get("strategy", ""))
            total_notional += amount
            market_exposure[market] = market_exposure.get(market, 0.0) + amount
            strategy_exposure[strategy] = strategy_exposure.get(strategy, 0.0) + amount
            if status in open_statuses:
                open_orders += 1

        return {
            "open_orders": open_orders,
            "market_exposure": market_exposure,
            "strategy_exposure": strategy_exposure,
            "total_notional": total_notional,
        }

    def intent_ids_by_order_id(self) -> dict[str, str]:
        order_intents: dict[str, str] = {}
        for record in self._records():
            if record.get("type") == "order_event" and record.get("order_id") and record.get("intent_id"):
                order_intents[str(record["order_id"])] = str(record["intent_id"])
        return order_intents

    def _records(self):
        if not os.path.exists(self.path):
            return

        with open(self.path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"corrupt order ledger at line {line_no}: {e}") from e
                yield record

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
