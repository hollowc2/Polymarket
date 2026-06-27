from datetime import UTC, datetime, timedelta
from io import BytesIO

import pandas as pd
from PIL import Image
from polymarket_algo.core.discord_trades import DiscordTrades


def test_chart_and_strategy_filter(monkeypatch, tmp_path):
    webhook = tmp_path / "webhook"
    webhook.write_text("https://example.invalid")
    monkeypatch.setenv("DISCORD_WEBHOOK_FILE", str(webhook))
    monkeypatch.setenv("DISCORD_STRATEGIES", "other")
    assert DiscordTrades(str(tmp_path / "trades.json"), "impulse_momentum").enabled is False

    now = datetime.now(UTC)
    frame = pd.DataFrame(
        {"open_time": [now - timedelta(minutes=5), now], "close": [100.0, 105.0]}
    )
    notifier = DiscordTrades(str(tmp_path / "trades.json"), "other")
    assert notifier.chart(frame, "test", int(now.timestamp() * 1000)).startswith(b"\x89PNG")


def test_chart_price_panel_uses_left_rail(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCORD_STRATEGIES", "other")
    now = datetime.now(UTC)
    frame = pd.DataFrame(
        {
            "open_time": [now - timedelta(minutes=5), now],
            "open": [100.0, 104.0],
            "high": [101.0, 106.0],
            "low": [99.0, 103.0],
            "close": [100.0, 105.0],
        }
    )
    chart = DiscordTrades(str(tmp_path / "trades.json"), "other").chart(
        frame,
        "test",
        int(now.timestamp() * 1000),
        yes_value=0.72,
        no_value=0.28,
    )

    image = Image.open(BytesIO(chart))
    assert image.getpixel((60, 65)) == (15, 23, 42)
