"""Pure analytics functions on list[TradeRecord]. No charts, no I/O."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from deepanal.models import TradeRecord


def summary(trades: list[TradeRecord]) -> dict:
    """High-level metrics dict: trade_count, win_rate, total_pnl, sharpe, max_drawdown."""
    settled = [t for t in trades if t.won is not None]
    if not settled:
        return {"trade_count": 0, "win_rate": 0.0, "total_pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    wins = sum(1 for t in settled if t.won)
    pnls = np.array([t.pnl for t in settled])
    equity = pd.Series(pnls).cumsum()
    max_dd = float((equity - equity.cummax()).min())
    std = float(pnls.std(ddof=0))
    sharpe = float(pnls.mean() / std * math.sqrt(len(pnls))) if std > 0 else 0.0

    return {
        "trade_count": len(settled),
        "win_rate": wins / len(settled),
        "total_pnl": float(pnls.sum()),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def equity_curve(trades: list[TradeRecord]) -> pd.Series:
    """Cumulative PnL series indexed by open_time (settled trades only)."""
    settled = sorted([t for t in trades if t.won is not None], key=lambda t: t.open_time)
    if not settled:
        return pd.Series(dtype=float)
    return pd.Series(
        [t.pnl for t in settled],
        index=[t.open_time for t in settled],
        name="equity",
    ).cumsum()


def win_rate_by_hour(trades: list[TradeRecord]) -> pd.Series:
    """Win rate keyed by UTC hour (0–23)."""
    settled = [t for t in trades if t.won is not None and t.hour_utc is not None]
    if not settled:
        return pd.Series(dtype=float)
    df = pd.DataFrame({"hour": [t.hour_utc for t in settled], "won": [t.won for t in settled]})
    return df.groupby("hour")["won"].mean().rename("win_rate")


def win_rate_by_weekday(trades: list[TradeRecord]) -> pd.Series:
    """Win rate keyed by day of week (0=Mon … 6=Sun)."""
    settled = [t for t in trades if t.won is not None and t.day_of_week is not None]
    if not settled:
        return pd.Series(dtype=float)
    df = pd.DataFrame({"dow": [t.day_of_week for t in settled], "won": [t.won for t in settled]})
    return df.groupby("dow")["won"].mean().rename("win_rate")


def pnl_by_hour(trades: list[TradeRecord]) -> pd.DataFrame:
    """Per-hour: win_rate, total_pnl, trade_count."""
    settled = [t for t in trades if t.won is not None and t.hour_utc is not None]
    if not settled:
        return pd.DataFrame(columns=["win_rate", "total_pnl", "trade_count"])
    df = pd.DataFrame({
        "hour": [t.hour_utc for t in settled],
        "won": [t.won for t in settled],
        "pnl": [t.pnl for t in settled],
    })
    return df.groupby("hour").agg(
        win_rate=("won", "mean"),
        total_pnl=("pnl", "sum"),
        trade_count=("won", "count"),
    )


def pnl_by_weekday(trades: list[TradeRecord]) -> pd.DataFrame:
    """Per-weekday: win_rate, total_pnl, trade_count."""
    settled = [t for t in trades if t.won is not None and t.day_of_week is not None]
    if not settled:
        return pd.DataFrame(columns=["win_rate", "total_pnl", "trade_count"])
    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df = pd.DataFrame({
        "dow": [t.day_of_week for t in settled],
        "won": [t.won for t in settled],
        "pnl": [t.pnl for t in settled],
    })
    result = df.groupby("dow").agg(
        win_rate=("won", "mean"),
        total_pnl=("pnl", "sum"),
        trade_count=("won", "count"),
    )
    result.index = result.index.map(day_names)
    return result


def pnl_by_gate(trades: list[TradeRecord]) -> pd.DataFrame:
    """Per gate+boosted combination: win_rate, total_pnl, trade_count (live only)."""
    settled = [t for t in trades if t.won is not None and t.source == "live"]
    if not settled:
        return pd.DataFrame()
    df = pd.DataFrame({
        "gate": [t.gate_name or "none" for t in settled],
        "boosted": [bool(t.gate_boosted) for t in settled],
        "won": [t.won for t in settled],
        "pnl": [t.pnl for t in settled],
    })
    return df.groupby(["gate", "boosted"]).agg(
        win_rate=("won", "mean"),
        total_pnl=("pnl", "sum"),
        trade_count=("won", "count"),
    ).reset_index()


def slippage_stats(trades: list[TradeRecord]) -> pd.DataFrame:
    """Per-trade slippage breakdown (live only, fill_price must be set)."""
    live = [t for t in trades if t.source == "live" and t.fill_price is not None]
    if not live:
        return pd.DataFrame()
    rows = []
    for t in live:
        overpay = (t.fill_price - t.best_ask) if t.best_ask else None  # type: ignore[operator]
        rows.append({
            "open_time": t.open_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "fill_price": t.fill_price,
            "slippage_pct": t.slippage_pct,
            "spread": t.spread,
            "overpay_vs_ask": overpay,
            "won": t.won,
            "pnl": t.pnl,
        })
    return pd.DataFrame(rows)


def streak_profile(trades: list[TradeRecord]) -> pd.DataFrame:
    """Win rate conditioned on N consecutive wins or losses before the trade."""
    settled = [t for t in trades if t.won is not None and t.consecutive_wins is not None]
    if not settled:
        return pd.DataFrame()
    rows = [
        {"streak_wins": t.consecutive_wins or 0, "streak_losses": t.consecutive_losses or 0, "won": t.won}
        for t in settled
    ]
    df = pd.DataFrame(rows)
    by_wins = (
        df.groupby("streak_wins")["won"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate", "count": "n"})
        .assign(streak_type="after_wins")
    )
    by_wins.index.name = "streak_len"
    by_losses = (
        df.groupby("streak_losses")["won"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate", "count": "n"})
        .assign(streak_type="after_losses")
    )
    by_losses.index.name = "streak_len"
    return pd.concat([by_wins.reset_index(), by_losses.reset_index()])


def hour_weekday_pivot(trades: list[TradeRecord], metric: str = "win_rate") -> pd.DataFrame:
    """Return a (7 × 24) pivot DataFrame for the heatmap (rows=weekday, cols=hour)."""
    settled = [t for t in trades if t.won is not None and t.hour_utc is not None and t.day_of_week is not None]
    if not settled:
        return pd.DataFrame(index=range(7), columns=range(24))
    df = pd.DataFrame({
        "hour": [t.hour_utc for t in settled],
        "dow": [t.day_of_week for t in settled],
        "won": [float(t.won) for t in settled],
        "pnl": [t.pnl for t in settled],
    })
    value_col = "won" if metric == "win_rate" else "pnl"
    pivot = df.pivot_table(index="dow", columns="hour", values=value_col, aggfunc="mean")
    return pivot.reindex(range(7)).reindex(columns=range(24))
