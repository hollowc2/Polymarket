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
    def __init__(self, *, response: dict | None = None, error: Exception | None = None) -> None:
        self.response = response if response is not None else {"orderID": "order-1"}
        self.error = error
        self.post_count = 0

    def create_market_order(self, order) -> object:
        return {"signed": order.kwargs}

    def post_order(self, signed_order, order_type) -> dict:
        self.post_count += 1
        if self.error is not None:
            raise self.error
        return self.response


class _FakeLedger:
    def __init__(
        self,
        *,
        intent_error: Exception | None = None,
        existing_intents: set[str] | None = None,
        read_error: Exception | None = None,
    ) -> None:
        self.intent_error = intent_error
        self.existing_intents = existing_intents or set()
        self.read_error = read_error
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
