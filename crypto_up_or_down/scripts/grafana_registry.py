"""Shared Grafana strategy registry helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_COMPOSE = _ROOT / "docker-compose.yml"
_BOT_RE = re.compile(r"([\w-]+-bot)")
_RETIRED_LINE_RE = re.compile(r"#\s*(?:RETIRED|Retired)\b")
_TOKEN_RE = re.compile(r"[\w-]+")


def strategy_from_bot_service(name: str) -> str:
    return name.removesuffix("-bot")


def _is_strategy_name(name: str) -> bool:
    if name in {"RETIRED", "Retired"} or name.isdigit():
        return False
    if re.fullmatch(r"\d+[mhd]", name):
        return False
    return "-" in name


def _expand_shorthand(token: str) -> list[str]:
    if "/" not in token:
        name = strategy_from_bot_service(token) if token.endswith("-bot") else token
        return [name] if _is_strategy_name(name) else []

    left, right = token.split("/", 1)
    if re.fullmatch(r"\d+[mhd]", right):
        base = re.sub(r"-\d+[mhd]$", "", left)
        names = [left, f"{base}-{right}"]
        return [name for name in names if _is_strategy_name(name)]
    return [token] if _is_strategy_name(token) else []


@lru_cache(maxsize=1)
def retired_strategies(compose_path: str | None = None) -> frozenset[str]:
    """Strategy names retired in docker-compose (commented-out bot services)."""
    path = Path(compose_path) if compose_path else _COMPOSE
    if not path.exists():
        return frozenset()

    out: set[str] = set()
    for line in path.read_text().splitlines():
        if not _RETIRED_LINE_RE.search(line):
            continue

        for bot in _BOT_RE.findall(line):
            out.add(strategy_from_bot_service(bot))

        payload = line.split(":", 1)[-1].split("—", 1)[0].split("#", 1)[0]
        for raw in _TOKEN_RE.findall(payload):
            for name in _expand_shorthand(raw):
                if _is_strategy_name(name):
                    out.add(name)

    return frozenset(out)
