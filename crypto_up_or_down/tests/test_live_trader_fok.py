"""LiveTrader FOK order safety tests."""

from __future__ import annotations

import json

import pytest
from polymarket_algo.core.config import Config
from polymarket_algo.executor.client import Market
from polymarket_algo.executor.order_ledger import JsonOrderLedger, OrderIntent, OrderLedgerEvent
from polymarket_algo.executor.trader import LiveTrader, Trade


class _FakeMarketOrderArgs:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeOrderType:
    FOK = "FOK"


class _FakeClient:
    def __init__(
        self,
        *,
        response: dict | None = None,
        error: Exception | None = None,
        orders: list[dict] | None = None,
        cancel_error: Exception | None = None,
    ) -> None:
        self.response = response if response is not None else {"orderID": "order-1"}
        self.error = error
        self.orders = orders or []
        self.cancel_error = cancel_error
        self.post_count = 0
        self.cancelled = []

    def create_market_order(self, order) -> object:
        return {"signed": order.kwargs}

    def post_order(self, signed_order, order_type) -> dict:
        self.post_count += 1
        if self.error is not None:
            raise self.error
        return self.response

    def get_orders(self):
        return self.orders

    def cancel(self, order_id: str) -> None:
        if self.cancel_error is not None:
            raise self.cancel_error
        self.cancelled.append(order_id)


class _FakeLedger:
    def __init__(
        self,
        *,
        intent_error: Exception | None = None,
        existing_intents: set[str] | None = None,
        read_error: Exception | None = None,
        risk_snapshot: dict | None = None,
    ) -> None:
        self.intent_error = intent_error
        self.existing_intents = existing_intents or set()
        self.read_error = read_error
        self._risk_snapshot = risk_snapshot or {
            "open_orders": 0,
            "market_exposure": {},
            "strategy_exposure": {},
            "total_notional": 0.0,
        }
        self.intents = []
        self.events = []

    def has_intent(self, intent_id: str) -> bool:
        if self.read_error is not None:
            raise self.read_error
        return intent_id in self.existing_intents

    def record_intent(self, intent) -> None:
        if self.intent_error is not None:
            raise self.intent_error
        self.intents.append(intent)

    def record_event(self, event) -> None:
        self.events.append(event)

    def risk_snapshot(self) -> dict:
        if self.read_error is not None:
            raise self.read_error
        return self._risk_snapshot

    def intent_ids_by_order_id(self) -> dict[str, str]:
        if self.read_error is not None:
            raise self.read_error
        return {}


def _market() -> Market:
    return Market(
        timestamp=1_771_051_500,
        slug="btc-updown-5m-1771051500",
        title="BTC Up or Down",
        closed=False,
        outcome=None,
        up_token_id="up-token",
        down_token_id="down-token",
        up_price=0.60,
        down_price=0.40,
        volume=1000.0,
        accepting_orders=True,
        taker_fee_bps=700,
    )


def _trader(
    *,
    client: _FakeClient | None = None,
    ledger: _FakeLedger | None = None,
    status_result: dict | None = None,
) -> LiveTrader:
    trader = LiveTrader.__new__(LiveTrader)
    trader.client = client if client is not None else _FakeClient()
    trader._order_ledger = ledger if ledger is not None else _FakeLedger()
    trader.MarketOrderArgs = _FakeMarketOrderArgs
    trader.OrderType = _FakeOrderType
    trader.BUY = "BUY"
    trader.SELL = "SELL"
    trader._startup_reconciliation_ok = True
    trader._get_order_status = lambda _order_id: status_result or {
        "status": "filled",
        "filled_size": 10.0,
        "avg_price": 0.50,
        "order": {},
    }
    return trader


@pytest.mark.parametrize(
    "status_result",
    [
        {"status": "unknown", "filled_size": 0.0, "avg_price": 0.0, "order": None},
        {"status": "cancelled", "filled_size": 0.0, "avg_price": 0.0, "order": {}},
    ],
)
def test_live_fok_unfilled_status_returns_none(status_result: dict) -> None:
    ledger = _FakeLedger()
    trader = _trader(ledger=ledger, status_result=status_result)

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert len(ledger.intents) == 1
    assert ledger.events[-1].event in {"order_unknown", "order_cancelled"}


def test_live_fok_post_exception_returns_none() -> None:
    ledger = _FakeLedger()
    trader = _trader(client=_FakeClient(error=TimeoutError("post timed out")), ledger=ledger)

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert len(ledger.intents) == 1
    assert ledger.events[-1].event == "order_failed"


@pytest.mark.parametrize(
    "response",
    [
        {"success": False, "error": "rejected"},
        {"error": "bad request"},
        {},
    ],
)
def test_live_fok_rejected_or_malformed_response_returns_none(response: dict) -> None:
    ledger = _FakeLedger()
    trader = _trader(client=_FakeClient(response=response), ledger=ledger)

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert len(ledger.intents) == 1
    assert ledger.events[-1].event == "order_rejected"


def test_live_fok_intent_persist_failure_returns_none_before_submit() -> None:
    client = _FakeClient()
    trader = _trader(client=client, ledger=_FakeLedger(intent_error=OSError("disk full")))

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0


def test_live_fok_duplicate_intent_returns_none_before_submit() -> None:
    client = _FakeClient()
    intent_id = "streak:btc-updown-5m-1771051500:up:1771051500"
    trader = _trader(client=client, ledger=_FakeLedger(existing_intents={intent_id}))

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0


def test_live_fok_corrupt_ledger_returns_none_before_submit() -> None:
    client = _FakeClient()
    trader = _trader(client=client, ledger=_FakeLedger(read_error=ValueError("corrupt ledger")))

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0


def test_live_fok_kill_switch_returns_none_before_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    ledger = _FakeLedger()
    trader = _trader(client=client, ledger=ledger)
    monkeypatch.setattr(Config, "LIVE_KILL_SWITCH", True)

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0
    assert ledger.intents == []


def test_live_fok_kill_file_returns_none_before_submit(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    kill_file = tmp_path / "kill"
    kill_file.write_text("stop", encoding="utf-8")
    client = _FakeClient()
    ledger = _FakeLedger()
    trader = _trader(client=client, ledger=ledger)
    monkeypatch.setattr(Config, "LIVE_KILL_SWITCH_FILE", str(kill_file))

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0
    assert ledger.intents == []


def test_live_fok_max_notional_returns_none_before_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    ledger = _FakeLedger()
    trader = _trader(client=client, ledger=ledger)
    monkeypatch.setattr(Config, "MAX_LIVE_ORDER_USD", 4.0)

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0
    assert ledger.intents == []


def test_live_fok_max_price_returns_none_before_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    ledger = _FakeLedger()
    trader = _trader(client=client, ledger=ledger)
    monkeypatch.setattr(Config, "MAX_LIVE_ORDER_PRICE", 0.55)

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0
    assert ledger.intents == []


def test_live_fok_startup_reconciliation_failure_returns_none_before_submit() -> None:
    client = _FakeClient()
    trader = _trader(client=client)
    trader._startup_reconciliation_ok = False

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0


def test_live_fok_stale_quote_returns_none_before_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    trader = _trader(client=client)
    monkeypatch.setattr(Config, "MAX_LIVE_QUOTE_AGE_SECONDS", 1.0)

    assert (
        trader.place_bet(
            _market(),
            "up",
            5.0,
            0.9,
            3,
            precomputed_execution={"fetched_at_ms": 1},
        )
        is None
    )
    assert client.post_count == 0


@pytest.mark.parametrize(
    ("config_name", "config_value", "snapshot"),
    [
        (
            "MAX_LIVE_OPEN_ORDERS",
            1,
            {"open_orders": 1, "market_exposure": {}, "strategy_exposure": {}, "total_notional": 0.0},
        ),
        (
            "MAX_LIVE_MARKET_EXPOSURE_USD",
            9.0,
            {
                "open_orders": 0,
                "market_exposure": {"btc-updown-5m-1771051500": 5.0},
                "strategy_exposure": {},
                "total_notional": 5.0,
            },
        ),
        (
            "MAX_LIVE_STRATEGY_EXPOSURE_USD",
            9.0,
            {"open_orders": 0, "market_exposure": {}, "strategy_exposure": {"streak": 5.0}, "total_notional": 5.0},
        ),
        (
            "MAX_LIVE_TOTAL_NOTIONAL_USD",
            9.0,
            {"open_orders": 0, "market_exposure": {}, "strategy_exposure": {}, "total_notional": 5.0},
        ),
    ],
)
def test_live_fok_exposure_caps_return_none_before_submit(
    config_name: str,
    config_value: float,
    snapshot: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    trader = _trader(client=client, ledger=_FakeLedger(risk_snapshot=snapshot))
    monkeypatch.setattr(Config, config_name, config_value)

    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0


def test_live_startup_reconciliation_cancels_open_orders() -> None:
    client = _FakeClient(orders=[{"id": "open-1", "status": "OPEN"}])
    ledger = _FakeLedger()
    trader = _trader(client=client, ledger=ledger)

    assert trader.reconcile_startup_orders() is True
    assert client.cancelled == ["open-1"]
    assert ledger.events[-1].event == "startup_order_cancelled"


def test_live_startup_reconciliation_records_recent_filled_orders() -> None:
    client = _FakeClient(orders=[{"id": "filled-1", "status": "FILLED"}])
    ledger = _FakeLedger()
    trader = _trader(client=client, ledger=ledger)

    assert trader.reconcile_startup_orders() is True
    assert client.cancelled == []
    assert ledger.events[-1].event == "startup_order_filled"
    assert ledger.events[-1].status == "filled"


def test_live_startup_reconciliation_failed_cancel_blocks_new_orders() -> None:
    client = _FakeClient(orders=[{"id": "open-1", "status": "OPEN"}], cancel_error=RuntimeError("nope"))
    ledger = _FakeLedger()
    trader = _trader(client=client, ledger=ledger)

    assert trader.reconcile_startup_orders() is False
    assert ledger.events[-1].event == "startup_order_cancel_failed"
    assert ledger.events[-1].status == "cancel_failed"
    trader._startup_reconciliation_ok = False
    assert trader.place_bet(_market(), "up", 5.0, 0.9, 3) is None
    assert client.post_count == 0


def test_live_fok_filled_status_returns_trade() -> None:
    ledger = _FakeLedger()
    trader = _trader(
        client=_FakeClient(response={"orderID": "order-1"}),
        ledger=ledger,
        status_result={
            "status": "filled",
            "filled_size": 12.5,
            "avg_price": 0.40,
            "order": {},
        },
    )

    trade = trader.place_bet(_market(), "down", 10.0, 0.9, 3, strategy="test")

    assert isinstance(trade, Trade)
    assert trade.paper is False
    assert trade.order_id == "order-1"
    assert trade.order_status == "filled"
    assert trade.amount == pytest.approx(5.0)
    assert trade.shares_bought == pytest.approx(12.5)
    assert trade.execution_price == pytest.approx(0.40)
    assert trade.requested_amount == pytest.approx(10.0)
    assert len(ledger.intents) == 1
    assert [event.event for event in ledger.events] == ["order_submitted", "order_filled"]


def test_json_order_ledger_appends_intent_and_events(tmp_path) -> None:
    ledger_path = tmp_path / "order-ledger.jsonl"
    ledger = JsonOrderLedger(str(ledger_path))
    intent = OrderIntent(
        id="test:market:up:1",
        strategy="test",
        market_slug="market",
        token_id="token",
        direction="up",
        side="BUY",
        amount_usd=5.0,
        max_price=0.60,
        created_at_ms=123,
    )

    ledger.record_intent(intent)
    ledger.record_event(
        OrderLedgerEvent(
            event="order_unknown",
            intent_id=intent.id,
            timestamp_ms=124,
            status="unknown",
            order_id="order-1",
            reason="status timeout",
        )
    )

    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["type"] == "order_intent"
    assert records[0]["intent"]["id"] == intent.id
    assert records[1]["type"] == "order_event"
    assert records[1]["event"] == "order_unknown"
    assert records[1]["status"] == "unknown"


def test_json_order_ledger_detects_existing_intent_and_corrupt_lines(tmp_path) -> None:
    ledger_path = tmp_path / "order-ledger.jsonl"
    ledger_path.write_text(
        '{"type":"order_intent","intent":{"id":"strategy:market:up:1"}}\n',
        encoding="utf-8",
    )

    ledger = JsonOrderLedger(str(ledger_path))
    assert ledger.has_intent("strategy:market:up:1") is True
    assert ledger.has_intent("strategy:market:down:1") is False

    ledger_path.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt order ledger"):
        ledger.has_intent("strategy:market:up:1")


def test_json_order_ledger_risk_snapshot_counts_open_and_exposure(tmp_path) -> None:
    ledger_path = tmp_path / "order-ledger.jsonl"
    ledger = JsonOrderLedger(str(ledger_path))

    for intent_id, market, strategy in [
        ("strategy:market-a:up:1", "market-a", "strategy"),
        ("strategy:market-b:up:1", "market-b", "strategy"),
    ]:
        ledger.record_intent(
            OrderIntent(
                id=intent_id,
                strategy=strategy,
                market_slug=market,
                token_id="token",
                direction="up",
                side="BUY",
                amount_usd=5.0,
                max_price=0.60,
                created_at_ms=123,
            )
        )
    ledger.record_event(
        OrderLedgerEvent(
            event="order_submitted",
            intent_id="strategy:market-a:up:1",
            timestamp_ms=124,
            status="submitted",
        )
    )
    ledger.record_event(
        OrderLedgerEvent(
            event="order_cancelled",
            intent_id="strategy:market-b:up:1",
            timestamp_ms=125,
            status="cancelled",
        )
    )
    ledger.record_intent(
        OrderIntent(
            id="strategy:market-c:up:1",
            strategy="strategy",
            market_slug="market-c",
            token_id="token",
            direction="up",
            side="BUY",
            amount_usd=5.0,
            max_price=0.60,
            created_at_ms=123,
        )
    )
    ledger.record_event(
        OrderLedgerEvent(
            event="order_filled",
            intent_id="strategy:market-c:up:1",
            timestamp_ms=126,
            status="filled",
        )
    )

    snapshot = ledger.risk_snapshot()

    assert snapshot["open_orders"] == 1
    assert snapshot["market_exposure"] == {"market-a": 5.0}
    assert snapshot["strategy_exposure"] == {"strategy": 5.0}
    assert snapshot["total_notional"] == 5.0


def test_json_order_ledger_maps_order_ids_to_intents(tmp_path) -> None:
    ledger_path = tmp_path / "order-ledger.jsonl"
    ledger = JsonOrderLedger(str(ledger_path))
    ledger.record_event(
        OrderLedgerEvent(
            event="order_submitted",
            intent_id="strategy:market:up:1",
            timestamp_ms=124,
            status="submitted",
            order_id="order-1",
        )
    )

    assert ledger.intent_ids_by_order_id() == {"order-1": "strategy:market:up:1"}
