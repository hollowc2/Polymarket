from __future__ import annotations

from scripts.grafana_archive import build_rows_from_history_records, parse_market_slug


def test_parse_market_slug_extracts_asset_and_timeframe() -> None:
    assert parse_market_slug("btc-updown-5m-foo") == ("btc", "5m")
    assert parse_market_slug("eth-updown-15m-bar") == ("eth", "15m")


def test_build_rows_preserves_raw_and_settled_history() -> None:
    records = [
        {
            "id": "trade-1",
            "market": {"timestamp": 1715000000, "slug": "btc-updown-5m-demo"},
            "position": {"direction": "up", "amount": 12.5},
            "execution": {
                "entry_price": 0.52,
                "fill_price": 0.51,
                "slippage_pct": 0.01,
            },
            "settlement": {"status": "settled", "won": True, "net_profit": 1.23, "outcome": "win"},
            "context": {"mode": "paper", "confidence": 0.7},
            "gate": {"name": "trend", "boosted": True},
            "session": {"trade_number": 4, "consecutive_wins": 2, "consecutive_losses": 0},
            "timing": {"hour_utc": 13, "day_of_week": 1},
        },
        {
            "id": "trade-2",
            "market": {"timestamp": 1715000300, "slug": "btc-updown-5m-demo"},
            "position": {"direction": "down", "amount": 10.0},
            "execution": {"entry_price": 0.49},
            "settlement": {"status": "pending", "won": None},
            "context": {"mode": "live"},
        },
    ]

    archive_rows, settled_rows = build_rows_from_history_records(
        records,
        strategy="demo-bot",
        source_file="demo-history.json",
    )

    assert len(archive_rows) == 2
    assert len(settled_rows) == 1

    first = archive_rows[0]
    assert first["strategy"] == "demo-bot"
    assert first["asset"] == "btc"
    assert first["timeframe"] == "5m"
    assert first["won"] is True
    assert first["paper"] is True
    assert first["raw_json"].adapted["id"] == "trade-1"

    second = archive_rows[1]
    assert second["won"] is None
    assert second["paper"] is False
    assert second["settlement_status"] == "pending"

