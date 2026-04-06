"""Load trade data from all sources into a unified list[TradeRecord]."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from deepanal.models import TradeRecord

if TYPE_CHECKING:
    from polymarket_algo.backtest.engine import BacktestResult

ROOT = Path(__file__).parents[2]

# Docker volume mount: /opt/polymarket/state → /app/state inside containers.
# Running outside containers (this app), the real path is /opt/polymarket/state.
STATE_DIR = Path("/opt/polymarket/state")
if not STATE_DIR.exists():
    STATE_DIR = ROOT / "state"  # fallback to project state/ if not on server


# ── public API ────────────────────────────────────────────────────────────────


def discover_bots(state_dir: Path | None = None) -> list[str]:
    """Return sorted list of bot names derived from *-history.json filenames.

    Each bot writes its own <bot-name>-history.json. The bot name is the
    stem before '-history', e.g. '3barmomo-5m-scale' from
    '3barmomo-5m-scale-history.json'.
    """
    d = state_dir or STATE_DIR
    bots = []
    for f in sorted(d.glob("*-history.json")):
        name = f.stem.removesuffix("-history")
        # Quick sanity-check: must be non-empty nested JSON list
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list) and data and "market" in data[0]:
                bots.append(name)
        except Exception:
            continue
    # Also offer the combined history file if it exists
    combined = d / "trade_history_full.json"
    if combined.exists():
        bots = ["[all]"] + bots
    return bots


def load_bot_trades(bot_name: str, state_dir: Path | None = None) -> list[TradeRecord]:
    """Load trades for a specific bot by name.

    Args:
        bot_name: e.g. '3barmomo-5m-scale' or '[all]' for the combined file.
        state_dir: override the default state directory.
    """
    d = state_dir or STATE_DIR
    if bot_name == "[all]":
        path = d / "trade_history_full.json"
    else:
        path = d / f"{bot_name}-history.json"
    return _load_file(path, bot_name)


def load_live_trades(
    path: Path | None = None,
    strategy_filter: str | None = None,
) -> list[TradeRecord]:
    """Load from an explicit file path (nested JSON format).

    Args:
        path: path to a *-history.json or trade_history_full.json
        strategy_filter: if set, only return trades for this strategy name
    """
    p = path or (STATE_DIR / "trade_history_full.json")
    trades = _load_file(p)
    if strategy_filter:
        trades = [t for t in trades if t.strategy == strategy_filter]
    return trades


def from_backtest_result(result: BacktestResult, strategy_name: str) -> list[TradeRecord]:
    """Convert a BacktestResult into a list[TradeRecord].

    signal == 1 → "up", signal == -1 → "down".
    The timestamp index of the trades DataFrame is the candle open_time,
    so it aligns directly to OHLCV.
    """
    import pandas as pd

    records: list[TradeRecord] = []
    for i, (ts, row) in enumerate(result.trades.iterrows()):
        direction: str = "up" if int(row["signal"]) == 1 else "down"
        ts_dt: datetime = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.fromtimestamp(float(ts), tz=UTC)
        records.append(
            TradeRecord(
                id=f"bt_{strategy_name}_{i}",
                strategy=strategy_name,
                source="backtest",
                open_time=ts_dt,
                executed_at=None,
                direction=direction,  # type: ignore[arg-type]
                amount=float(row["size"]),
                entry_price=float(row["entry_close"]),
                fill_price=None,
                won=bool(row["is_win"]),
                pnl=float(row["pnl"]),
                hour_utc=ts_dt.hour,
                day_of_week=ts_dt.weekday(),
            )
        )
    return records


# ── internal ──────────────────────────────────────────────────────────────────


def _load_file(path: Path, bot_name: str | None = None) -> list[TradeRecord]:
    if not path.exists():
        return []
    with path.open() as f:
        raw: list[dict] = json.load(f)

    trades: list[TradeRecord] = []
    for r in raw:
        try:
            t = _parse_live(r)
            # Stamp the bot name onto strategy so UI can distinguish bots
            # that all share context.strategy='streak'
            if bot_name and bot_name != "[all]":
                t.strategy = bot_name
            trades.append(t)
        except Exception:
            continue
    return sorted(trades, key=lambda t: t.open_time)


def _parse_live(r: dict) -> TradeRecord:
    market = r.get("market", {})
    position = r.get("position", {})
    execution = r.get("execution", {})
    settlement = r.get("settlement", {})
    context = r.get("context", {})
    session = r.get("session", {})
    timing = r.get("timing", {})
    gate = r.get("gate", {})

    # open_time: market.timestamp is epoch seconds (window open → candle open_time)
    ts_s = market.get("timestamp", 0)
    open_time = datetime.fromtimestamp(ts_s, tz=UTC)

    # executed_at: execution.timestamp is epoch milliseconds
    exec_ms = execution.get("timestamp")
    executed_at = datetime.fromtimestamp(exec_ms / 1000, tz=UTC) if exec_ms else None

    direction = str(position.get("direction", "up")).lower()
    if direction not in ("up", "down"):
        direction = "up"

    raw_fill = execution.get("fill_price", 0.0)
    fill_price = float(raw_fill) if raw_fill else None

    return TradeRecord(
        id=r.get("id", str(uuid.uuid4())),
        strategy=context.get("strategy", "unknown"),
        source="live",
        open_time=open_time,
        executed_at=executed_at,
        direction=direction,  # type: ignore[arg-type]
        amount=float(position.get("amount", 0.0)),
        entry_price=float(execution.get("entry_price", 0.5)),
        fill_price=fill_price,
        won=settlement.get("won"),
        pnl=float(settlement.get("net_profit", 0.0)),
        gate_name=gate.get("name") or None,
        gate_boosted=gate.get("boosted"),
        slippage_pct=execution.get("slippage_pct"),
        spread=execution.get("spread"),
        fill_pct=execution.get("fill_pct"),
        best_bid=execution.get("best_bid"),
        best_ask=execution.get("best_ask"),
        price_movement_pct=execution.get("price_movement_pct"),
        session_trade_n=session.get("trade_number"),
        hour_utc=timing.get("hour_utc"),
        day_of_week=timing.get("day_of_week"),
        consecutive_wins=session.get("consecutive_wins"),
        consecutive_losses=session.get("consecutive_losses"),
        bankroll_before=session.get("bankroll_before"),
        market_bias=context.get("market_bias"),
        is_paper=context.get("mode", "paper") == "paper",
        market_slug=market.get("slug"),
    )
