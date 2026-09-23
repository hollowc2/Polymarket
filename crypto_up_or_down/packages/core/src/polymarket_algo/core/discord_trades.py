"""Small Discord webhook sender for confirmed trade fills."""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageDraw


class DiscordTrades:
    def __init__(self, state_file: str, strategy: str) -> None:
        selected = {item.strip() for item in os.getenv("DISCORD_STRATEGIES", "impulse_momentum").split(",")}
        self.enabled = strategy in selected
        webhook_file = Path(os.getenv("DISCORD_WEBHOOK_FILE", "/run/secrets/discord-webhook"))
        self.webhook = webhook_file.read_text().strip() if self.enabled and webhook_file.exists() else ""
        self.links_file = Path(state_file).with_name(f"{strategy}-discord.json")

    def chart(
        self,
        frame,
        title: str,
        entry_ms: int,
        exit_ms: int | None = None,
        *,
        highlight_ms: tuple[int, int] | None = None,
        subtitle: str = "",
        yes_label: str = "YES",
        yes_value: float | None = None,
        no_label: str = "NO",
        no_value: float | None = None,
    ) -> bytes | None:
        if frame is None or len(frame) < 2:
            return None
        values = [float(value) for value in frame["close"]]
        highs = [float(value) for value in frame["high"]] if "high" in frame else values
        lows = [float(value) for value in frame["low"]] if "low" in frame else values
        opens = [float(value) for value in frame["open"]] if "open" in frame else values
        width, height, pad = 900, 420, 55
        has_price_panel = yes_value is not None or no_value is not None
        panel_width = 210 if has_price_panel else 0
        plot_left = pad + panel_width + (25 if has_price_panel else 0)
        plot_right = width - pad
        image = Image.new("RGB", (width, height), "#111827")
        draw = ImageDraw.Draw(image)
        low, high = min(lows), max(highs)
        span = high - low or 1.0

        def x_at(index: int) -> float:
            return plot_left + index * (plot_right - plot_left) / (len(values) - 1)

        def y_at(value: float) -> float:
            return height - pad - (value - low) * (height - 2 * pad) / span

        times = [int(ts.timestamp() * 1000) for ts in frame["open_time"]]
        if highlight_ms:
            start, end = highlight_ms
            left = x_at(min(range(len(times)), key=lambda i: abs(times[i] - start)))
            right = x_at(min(range(len(times)), key=lambda i: abs(times[i] - end)))
            if right <= left:
                right = left + (plot_right - plot_left) / max(len(values) - 1, 1)
            draw.rectangle((left, pad - 12, right, height - pad), fill="#1f2937")

        candle_width = max(4, min(16, int((plot_right - plot_left) / len(values) * 0.55)))
        for index, (open_, high_, low_, close_) in enumerate(zip(opens, highs, lows, values, strict=True)):
            x = x_at(index)
            color = "#22c55e" if close_ >= open_ else "#ef4444"
            draw.line((x, y_at(low_), x, y_at(high_)), fill=color, width=2)
            top, bottom = sorted((y_at(open_), y_at(close_)))
            if bottom <= top:
                bottom = top + 1
            draw.rectangle((x - candle_width / 2, top, x + candle_width / 2, bottom), fill=color)

        draw.text((plot_left, 18), title, fill="white")
        if subtitle:
            draw.text((plot_left, 38), subtitle, fill="#cbd5e1")
        draw.text((plot_left, height - 35), f"{low:,.2f}", fill="#94a3b8")
        draw.text((width - 140, 18), f"{high:,.2f}", fill="#94a3b8")
        if has_price_panel:
            draw.rectangle((pad, 58, pad + panel_width, 132), fill="#0f172a", outline="#334155", width=1)
            if yes_value is not None:
                draw.text((pad + 20, 76), f"{yes_label}: {yes_value:.3f}", fill="#86efac")
            if no_value is not None:
                draw.text((pad + 20, 104), f"{no_label}: {no_value:.3f}", fill="#fca5a5")
        markers = [(entry_ms, "#facc15", "ENTRY"), (exit_ms, "#c084fc", "EXIT")]
        marker_indexes = [
            min(range(len(times)), key=lambda i: abs(times[i] - timestamp)) if timestamp is not None else None
            for timestamp, _color, _label in markers
        ]
        overlap = marker_indexes[0] is not None and marker_indexes[0] == marker_indexes[1]
        for offset, (timestamp, color, label) in zip((-18, 8) if overlap else (0, 0), markers, strict=True):
            if timestamp is None:
                continue
            index = min(range(len(times)), key=lambda i: abs(times[i] - timestamp))
            x, y = x_at(index), y_at(values[index])
            draw.line((x, pad - 12, x, height - pad), fill=color, width=1)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
            draw.text((x + 8, y + offset), label, fill=color)
        for index in (0, len(times) - 1):
            label = frame["open_time"].iloc[index].strftime("%H:%M")
            draw.text((x_at(index) - 18, height - 20), label, fill="#94a3b8")
        output = io.BytesIO()
        image.save(output, "PNG")
        return output.getvalue()

    def send(self, key: str, content: str, chart: bytes | None = None, *, remember: bool = False) -> None:
        if not self.webhook:
            return
        link = self._load().get(key)
        if link and not remember:
            content += f"\n> Entry: {link}"
        payload = {"content": content, "allowed_mentions": {"parse": []}}
        url = f"{self.webhook}?wait=true"
        if chart:
            boundary = uuid.uuid4().hex
            body = (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\n"
                "Content-Type: application/json\r\n\r\n"
                f"{json.dumps(payload)}\r\n--{boundary}\r\n"
                "Content-Disposition: form-data; name=\"files[0]\"; filename=\"trade.png\"\r\n"
                "Content-Type: image/png\r\n\r\n"
            ).encode() + chart + f"\r\n--{boundary}--\r\n".encode()
            request = urllib.request.Request(
                url,
                body,
                {
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "CryptoUpDown/1.0",
                },
            )
        else:
            request = urllib.request.Request(
                url,
                json.dumps(payload).encode(),
                {"Content-Type": "application/json", "User-Agent": "CryptoUpDown/1.0"},
            )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                message = json.load(response)
            if remember:
                links = self._load()
                guild_id = message.get("guild_id")
                if not guild_id:
                    request = urllib.request.Request(
                        self.webhook, headers={"User-Agent": "CryptoUpDown/1.0"}
                    )
                    with urllib.request.urlopen(request, timeout=10) as response:
                        guild_id = json.load(response)["guild_id"]
                links[key] = (
                    f"https://discord.com/channels/{guild_id}/"
                    f"{message['channel_id']}/{message['id']}"
                )
                self.links_file.write_text(json.dumps(links))
        except Exception as exc:
            print(f"Discord notification failed: {exc}", file=sys.stderr, flush=True)

    def _load(self) -> dict[str, str]:
        try:
            return json.loads(self.links_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
