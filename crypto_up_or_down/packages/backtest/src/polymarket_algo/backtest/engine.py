from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, TypeGuard, cast

import numpy as np
import pandas as pd
from polymarket_algo.core import Strategy

StrategyCallable = Callable[..., pd.Series | pd.DataFrame]
StrategyLike = Strategy | StrategyCallable


def _has_evaluate(strategy: StrategyLike) -> TypeGuard[Strategy]:
    """Return True if strategy is an object with a callable .evaluate() method."""
    return callable(getattr(strategy, "evaluate", None))


@dataclass
class BacktestResult:
    metrics: dict[str, Any]
    trades: pd.DataFrame
    pnl_curve: pd.Series


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _evaluate_strategy_output(
    candles: pd.DataFrame,
    strategy: StrategyLike,
    strategy_params: dict[str, Any],
) -> pd.Series | pd.DataFrame:
    if _has_evaluate(strategy):
        return strategy.evaluate(candles, **strategy_params)
    return cast(StrategyCallable, strategy)(candles, **strategy_params)


def _spike_candle_mask(
    candles: pd.DataFrame,
    atr_window: int,
    spike_atr_mult: float,
) -> pd.Series:
    """Boolean mask (shifted to the *next* candle) that is True when the next
    candle's high-low range exceeds ``spike_atr_mult × rolling_atr``.

    ATR proxy = rolling mean of bar ranges, computed on *current* candles only
    (no lookahead).
    """
    bar_range = candles["high"] - candles["low"]
    rolling_atr = bar_range.rolling(atr_window, min_periods=1).mean()
    next_range = bar_range.shift(-1)
    return next_range > spike_atr_mult * rolling_atr


def score_resolution(
    entry_close: pd.Series,
    next_high: pd.Series,
    next_low: pd.Series,
    next_close: pd.Series,
    signals: pd.Series,
    mode: str,
    win_payout: float,
    buy_price: float,
    half_spread: float = 0.0,
    spike_candle_mask: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Score trades under a given resolution model.

    Parameters
    ----------
    mode:
        ``"close"`` – standard close-to-close proxy (current behaviour).
        ``"intrabar_conservative"`` – on *spike candles only*, marks a win
        as a loss when the intrabar reversal shadow exceeds the net
        directional move.  Normal candles are scored identically to
        ``"close"``.  Pass a precomputed ``spike_candle_mask`` (True where
        the resolution candle is a spike) to restrict the haircut.  If the
        mask is None the haircut applies to *all* candles (original,
        overly-aggressive behaviour).
    spike_candle_mask:
        Boolean Series aligned to ``entry_close.index``.  True where the
        *next* resolution candle qualifies as a spike candle.  Only
        consulted in ``"intrabar_conservative"`` mode.

    Returns
    -------
    (outcome_up, per_share_pnl) both as Series aligned to entry_close.index
    """
    outcome_up = (next_close > entry_close).astype(int)
    direction_up = signals == 1
    wins = (direction_up & (outcome_up == 1)) | ((signals == -1) & (outcome_up == 0))

    if mode == "intrabar_conservative":
        # Net move in the direction of the bet (always ≥ 0 for a proxy-win)
        net_move = (next_close - entry_close).abs()

        # Shadow opposing the directional move:
        #   UP bet  → upper shadow = high − close
        #   DOWN bet → lower shadow = close − low
        upper_shadow = (next_high - next_close).clip(lower=0.0)
        lower_shadow = (next_close - next_low).clip(lower=0.0)

        # Contested: reversal shadow exceeds net move (candle whipsawed more
        # than it sustained direction — oracle close reflects the reversal).
        contested = (direction_up & (outcome_up == 1) & (upper_shadow > net_move)) | (
            (signals == -1) & (outcome_up == 0) & (lower_shadow > net_move)
        )

        # Restrict haircut to genuine spike candles; on normal candles the
        # close-to-close proxy is accurate enough.
        if spike_candle_mask is not None:
            contested = contested & spike_candle_mask

        wins = wins & ~contested

    effective_buy = buy_price + half_spread
    per_share_pnl = pd.Series(
        np.where(wins, win_payout - effective_buy, -effective_buy),
        index=entry_close.index,
    )
    return outcome_up, per_share_pnl


def run_backtest(
    candles: pd.DataFrame,
    strategy: StrategyLike,
    strategy_params: dict[str, Any] | None = None,
    buy_price: float = 0.50,
    win_payout: float = 0.95,
    resolution_mode: str = "close",
    half_spread: float = 0.0,
    atr_window: int = 20,
    spike_atr_mult: float = 2.5,
) -> BacktestResult:
    """Run a single backtest.

    Parameters
    ----------
    resolution_mode:
        ``"close"`` (default) — standard close-to-close proxy.
        ``"intrabar_conservative"`` — applies a shadow haircut on spike
        candles only.  Requires ``high`` and ``low`` columns in *candles*.
    atr_window:
        Rolling window (bars) for the ATR used to detect spike candles.
        Default 20 (= 100 min at 5m).
    spike_atr_mult:
        A resolution candle is flagged as a spike when its range exceeds
        ``spike_atr_mult × rolling_atr``.  Default 2.5.
    """
    strategy_params = strategy_params or {}

    out = _evaluate_strategy_output(candles, strategy, strategy_params)
    if isinstance(out, pd.DataFrame):
        signals = out["signal"].astype(int)
        size = out["size"].astype(float) if "size" in out.columns else pd.Series(15.0, index=candles.index)
    else:
        signals = out.astype(int)
        size = pd.Series(15.0, index=candles.index)

    next_close = candles["close"].shift(-1)
    next_high = candles["high"].shift(-1) if "high" in candles.columns else next_close
    next_low = candles["low"].shift(-1) if "low" in candles.columns else next_close

    mask: pd.Series | None = None
    if resolution_mode == "intrabar_conservative" and "high" in candles.columns and "low" in candles.columns:
        mask = _spike_candle_mask(candles, atr_window, spike_atr_mult)

    outcome_up, per_share_pnl = score_resolution(
        entry_close=candles["close"],
        next_high=next_high,
        next_low=next_low,
        next_close=next_close,
        signals=signals,
        mode=resolution_mode,
        win_payout=win_payout,
        buy_price=buy_price,
        half_spread=half_spread,
        spike_candle_mask=mask,
    )

    active = (signals != 0) & outcome_up.notna()
    wins = per_share_pnl > 0

    trade_pnl = (per_share_pnl * size).where(active, 0.0)
    pnl_curve = trade_pnl.cumsum()

    trades = pd.DataFrame(
        {
            "timestamp": candles.index,
            "signal": signals,
            "size": size,
            "entry_close": candles["close"],
            "next_close": next_close,
            "is_win": wins.where(active, False),
            "pnl": trade_pnl,
        }
    )
    trades = trades.loc[active]

    trade_count = int(active.sum())
    win_rate = float(trades["is_win"].mean()) if trade_count else 0.0
    total_pnl = float(trade_pnl.sum())
    returns = trade_pnl.loc[active]
    sharpe = (
        float((returns.mean() / returns.std(ddof=0)) * np.sqrt(len(returns)))
        if trade_count and returns.std(ddof=0) > 0
        else 0.0
    )

    metrics = {
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown": _max_drawdown(pnl_curve),
        "sharpe_ratio": sharpe,
        "trade_count": trade_count,
    }

    return BacktestResult(metrics=metrics, trades=trades, pnl_curve=pnl_curve)


def parameter_sweep(
    candles: pd.DataFrame,
    strategy: StrategyLike,
    param_grid: dict[str, list[Any]],
    resolution_mode: str = "close",
    half_spread: float = 0.0,
    atr_window: int = 20,
    spike_atr_mult: float = 2.5,
) -> pd.DataFrame:
    keys = list(param_grid.keys())
    rows: list[dict[str, Any]] = []

    for values in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values, strict=False))
        result = run_backtest(
            candles,
            strategy,
            params,
            resolution_mode=resolution_mode,
            half_spread=half_spread,
            atr_window=atr_window,
            spike_atr_mult=spike_atr_mult,
        )
        rows.append({**params, **result.metrics})

    return pd.DataFrame(rows).sort_values(by=["win_rate", "total_pnl"], ascending=False).reset_index(drop=True)


def walk_forward_split(candles: pd.DataFrame, train_ratio: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(candles) * train_ratio)
    train = candles.iloc[:split_idx].copy()
    test = candles.iloc[split_idx:].copy()
    return train, test
