from datetime import UTC, datetime, timedelta

import pandas as pd
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
