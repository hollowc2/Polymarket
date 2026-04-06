"""Plotly chart builders. Every function returns a go.Figure."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from deepanal import metrics as m
from deepanal.models import TradeRecord

_DARK = "plotly_dark"
_GREEN = "#26a69a"
_RED = "#ef5350"
_WIN = "#00e676"
_LOSS = "#ff5252"
_PENDING = "#888888"
_GOLD = "#ffd700"


# ── main chart ────────────────────────────────────────────────────────────────


def candlestick_trades(
    ohlcv: pd.DataFrame,
    trades: list[TradeRecord],
    show_volume: bool = True,
    show_equity: bool = True,
) -> go.Figure:
    """Candlestick OHLCV with trade entry markers, optional volume and equity subplots."""
    n_rows = 1 + int(show_volume) + int(show_equity)
    row_heights = [0.60] + [0.20] * (n_rows - 1)
    subplot_titles: list[str] = ["Price"]
    if show_volume:
        subplot_titles.append("Volume")
    if show_equity:
        subplot_titles.append("Equity")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["open"],
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            name="OHLCV",
            increasing_line_color=_GREEN,
            decreasing_line_color=_RED,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    _add_trade_markers(fig, trades, row=1, ohlcv=ohlcv)

    # Volume bars
    if show_volume:
        bar_colors = [
            _GREEN if float(c) >= float(o) else _RED
            for o, c in zip(ohlcv["open"], ohlcv["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=ohlcv.index,
                y=ohlcv["volume"],
                name="Volume",
                marker_color=bar_colors,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Equity curve
    if show_equity:
        eq_row = 2 + int(show_volume)
        eq = m.equity_curve(trades)
        if not eq.empty:
            fig.add_trace(
                go.Scatter(
                    x=eq.index,
                    y=eq.values,
                    name="Equity",
                    line={"color": _GOLD, "width": 1.5},
                    fill="tozeroy",
                    fillcolor="rgba(255,215,0,0.08)",
                ),
                row=eq_row,
                col=1,
            )

    # Clamp x-axis to the OHLCV window so equity/marker traces outside the
    # candle range don't shrink the visible candlebars.
    x_min = ohlcv.index.min()
    x_max = ohlcv.index.max()
    fig.update_layout(
        template=_DARK,
        height=720,
        margin={"l": 50, "r": 30, "t": 40, "b": 30},
        xaxis_rangeslider_visible=False,
        xaxis_range=[x_min, x_max],
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(showgrid=False)
    return fig


def _snap_to_ohlcv(trade: TradeRecord, ohlcv: pd.DataFrame) -> float:
    """Return the OHLCV candle price nearest to the trade entry time.

    Places UP entries below the candle low (for visibility) and DOWN entries
    above the candle high.  Falls back to the candle close if index lookup fails.
    """
    idx = ohlcv.index
    pos = idx.searchsorted(trade.open_time)
    pos = min(pos, len(idx) - 1)
    row = ohlcv.iloc[pos]
    if trade.direction == "up":
        return float(row["low"]) * 0.9995
    return float(row["high"]) * 1.0005


def _add_trade_markers(
    fig: go.Figure,
    trades: list[TradeRecord],
    row: int,
    ohlcv: pd.DataFrame | None = None,
) -> None:
    """Add triangular markers for each trade, grouped by direction × outcome.

    If *ohlcv* is supplied the markers are snapped to the candle high/low so
    they sit just outside the wick — keeping the y-axis scaled to BTC prices
    rather than Polymarket probabilities (0–1).
    """
    groups = {
        ("up", True):   {"symbol": "triangle-up",        "color": _WIN,     "label": "UP win"},
        ("up", False):  {"symbol": "triangle-up-open",   "color": _LOSS,    "label": "UP loss"},
        ("up", None):   {"symbol": "triangle-up-open",   "color": _PENDING, "label": "UP pending"},
        ("down", True): {"symbol": "triangle-down",      "color": _WIN,     "label": "DOWN win"},
        ("down", False):{"symbol": "triangle-down-open", "color": _LOSS,    "label": "DOWN loss"},
        ("down", None): {"symbol": "triangle-down-open", "color": _PENDING, "label": "DOWN pending"},
    }

    buckets: dict[tuple, list[TradeRecord]] = {k: [] for k in groups}
    for t in trades:
        key = (t.direction, t.won)
        if key in buckets:
            buckets[key].append(t)

    for key, group in buckets.items():
        if not group:
            continue
        style = groups[key]
        if ohlcv is not None and not ohlcv.empty:
            y_vals = [_snap_to_ohlcv(t, ohlcv) for t in group]
        else:
            y_vals = [t.fill_price if t.fill_price else t.entry_price for t in group]
        sizes = [max(8, min(24, 8 + t.amount / 5)) for t in group]
        hover = [
            (
                f"<b>{t.direction.upper()}</b> ${t.amount:.2f}<br>"
                f"Polymarket price: {t.fill_price or t.entry_price:.4f}<br>"
                f"PnL: ${t.pnl:+.2f}<br>"
                f"Strategy: {t.strategy}<br>"
                f"{t.open_time.strftime('%Y-%m-%d %H:%M UTC')}"
            )
            for t in group
        ]
        fig.add_trace(
            go.Scatter(
                x=[t.open_time for t in group],
                y=y_vals,
                mode="markers",
                name=style["label"],
                marker={
                    "symbol": style["symbol"],
                    "color": style["color"],
                    "size": sizes,
                    "line": {"width": 1.5, "color": style["color"]},
                },
                text=hover,
                hoverinfo="text",
            ),
            row=row,
            col=1,
        )


# ── per-trade mini chart ──────────────────────────────────────────────────────


def trade_detail_chart(
    trade: TradeRecord,
    ohlcv: pd.DataFrame,
    context_candles: int = 30,
) -> go.Figure:
    """Clipped candlestick ±context_candles around a single trade entry.

    Shows the entry marker, a vertical line at entry time, and an annotation
    with direction / PnL / outcome.
    """
    idx = ohlcv.index
    entry_pos = idx.searchsorted(trade.open_time)
    lo = max(0, entry_pos - context_candles)
    hi = min(len(idx), entry_pos + context_candles + 1)
    window = ohlcv.iloc[lo:hi]

    if window.empty:
        return _empty(f"No OHLCV data around {trade.open_time:%Y-%m-%d %H:%M}")

    # Outcome styling
    if trade.won is True:
        outcome_color = _WIN
        outcome_label = "WIN"
    elif trade.won is False:
        outcome_color = _LOSS
        outcome_label = "LOSS"
    else:
        outcome_color = _PENDING
        outcome_label = "PENDING"

    bar_colors = [
        _GREEN if float(c) >= float(o) else _RED
        for o, c in zip(window["open"], window["close"])
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=window.index,
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            increasing_line_color=_GREEN,
            decreasing_line_color=_RED,
            showlegend=False,
            name="Price",
        ),
        row=1, col=1,
    )

    # Signal marker — snap to candle high/low so the triangle sits outside the wick.
    # (entry_price / fill_price are Polymarket binary probabilities, 0–1, not BTC prices.)
    marker_y = _snap_to_ohlcv(trade, window)
    poly_price = trade.fill_price if trade.fill_price else trade.entry_price
    marker_symbol = "triangle-up" if trade.direction == "up" else "triangle-down"
    fig.add_trace(
        go.Scatter(
            x=[trade.open_time],
            y=[marker_y],
            mode="markers+text",
            marker={
                "symbol": marker_symbol,
                "color": outcome_color,
                "size": 16,
                "line": {"width": 2, "color": outcome_color},
            },
            text=[f"  {trade.direction.upper()}"],
            textposition="middle right",
            textfont={"color": outcome_color, "size": 11},
            showlegend=False,
            name="Signal",
        ),
        row=1, col=1,
    )

    # Vertical line at entry
    entry_ts = trade.open_time
    fig.add_vline(
        x=entry_ts.timestamp() * 1000,  # plotly expects ms for datetime axes
        line_dash="dot",
        line_color=outcome_color,
        line_width=1,
        opacity=0.6,
    )

    # ── Entry fill marker ────────────────────────────────────────────────────
    # For live trades use executed_at (actual order time); fall back to open_time.
    fill_ts = trade.executed_at if trade.executed_at else trade.open_time
    fill_pos = min(idx.searchsorted(fill_ts), len(idx) - 1)
    entry_fill_btc = float(ohlcv.iloc[fill_pos]["close"])
    fig.add_trace(
        go.Scatter(
            x=[idx[fill_pos]],
            y=[entry_fill_btc],
            mode="markers",
            marker={
                "symbol": "circle",
                "color": outcome_color,
                "size": 10,
                "line": {"width": 1.5, "color": "#000"},
            },
            name="Entry fill",
            showlegend=True,
            hovertemplate=(
                f"<b>Entry fill</b><br>"
                f"BTC: ${entry_fill_btc:,.2f}<br>"
                f"Poly price: {poly_price:.4f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # ── Exit fill marker ─────────────────────────────────────────────────────
    # Resolution is next-candle close (same logic as backtest engine).
    exit_pos = min(entry_pos + 1, len(idx) - 1)
    if exit_pos != entry_pos:  # guard: don't duplicate if at last candle
        exit_ts = idx[exit_pos]
        exit_fill_btc = float(ohlcv.iloc[exit_pos]["close"])
        exit_color = _WIN if trade.won is True else (_LOSS if trade.won is False else _PENDING)
        fig.add_trace(
            go.Scatter(
                x=[exit_ts],
                y=[exit_fill_btc],
                mode="markers",
                marker={
                    "symbol": "circle",
                    "color": exit_color,
                    "size": 10,
                    "line": {"width": 1.5, "color": "#000"},
                    "opacity": 0.75,
                },
                name="Exit fill",
                showlegend=True,
                hovertemplate=(
                    f"<b>Exit fill</b><br>"
                    f"BTC: ${exit_fill_btc:,.2f}<br>"
                    f"Move: ${exit_fill_btc - entry_fill_btc:+,.2f}"
                    f"  ({(exit_fill_btc/entry_fill_btc - 1)*100:+.2f}%)"
                    f"<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        # Connecting line between entry and exit fills
        fig.add_trace(
            go.Scatter(
                x=[idx[fill_pos], exit_ts],
                y=[entry_fill_btc, exit_fill_btc],
                mode="lines",
                line={"color": exit_color, "width": 1.5, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Volume bars
    if "volume" in window.columns:
        fig.add_trace(
            go.Bar(
                x=window.index,
                y=window["volume"],
                marker_color=bar_colors,
                showlegend=False,
                name="Volume",
            ),
            row=2, col=1,
        )

    direction_arrow = "▲" if trade.direction == "up" else "▼"
    pnl_str = f"${trade.pnl:+.2f}" if trade.pnl != 0 else "pending"
    title = (
        f"{direction_arrow} {trade.direction.upper()}  |  {outcome_label}  |  {pnl_str}  |  "
        f"${trade.amount:.2f}  @  {poly_price:.4f}  |  "
        f"{trade.open_time:%Y-%m-%d %H:%M UTC}"
    )

    fig.update_layout(
        template=_DARK,
        title={"text": title, "font": {"size": 13, "color": outcome_color}},
        height=400,
        margin={"l": 50, "r": 30, "t": 50, "b": 30},
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(showgrid=False)
    return fig


# ── standalone charts ─────────────────────────────────────────────────────────


def equity_curve_chart(trades: list[TradeRecord]) -> go.Figure:
    """Equity curve with drawdown shading."""
    eq = m.equity_curve(trades)
    if eq.empty:
        return _empty("No settled trades")

    peak = eq.cummax()
    dd = eq - peak

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.04,
        subplot_titles=["Cumulative PnL", "Drawdown"],
    )
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity", line={"color": _GOLD, "width": 2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=peak.index, y=peak.values, name="Peak", line={"color": "#555", "dash": "dot", "width": 1}, showlegend=False), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values, name="Drawdown",
            fill="tozeroy", fillcolor="rgba(255,82,82,0.2)",
            line={"color": _LOSS, "width": 1},
        ),
        row=2, col=1,
    )
    fig.update_layout(template=_DARK, height=500, margin={"l": 50, "r": 30, "t": 40, "b": 30})
    return fig


def heatmap_hour_weekday(trades: list[TradeRecord], metric: str = "win_rate") -> go.Figure:
    """24 × 7 heatmap of win rate or avg PnL per hour/weekday cell."""
    pivot = m.hour_weekday_pivot(trades, metric)
    if pivot.isnull().all().all():
        return _empty("Not enough data for heatmap")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hour_labels = [f"{h:02d}h" for h in range(24)]
    title = "Win Rate" if metric == "win_rate" else "Avg PnL"

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=hour_labels,
        y=day_labels,
        colorscale="RdYlGn",
        zmid=0.5 if metric == "win_rate" else 0,
        colorbar={"title": title},
        hoverongaps=False,
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>" + title + ": %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template=_DARK,
        title=f"{title} by Hour (UTC) × Weekday",
        height=380,
        margin={"l": 60, "r": 40, "t": 50, "b": 50},
    )
    return fig


def gate_breakdown_chart(trades: list[TradeRecord]) -> go.Figure:
    """Grouped bar: gate name × boosted → win rate."""
    df = m.pnl_by_gate(trades)
    if df.empty:
        return _empty("No gate data (live trades only)")

    fig = go.Figure()
    for boosted, color, label in [(True, _WIN, "Boosted"), (False, "#4fc3f7", "Normal")]:
        sub = df[df["boosted"] == boosted]
        if sub.empty:
            continue
        x_labels = [f"{row['gate']} (n={int(row['trade_count'])})" for _, row in sub.iterrows()]
        fig.add_trace(go.Bar(
            name=label,
            x=x_labels,
            y=sub["win_rate"].values,
            text=[f"{v:.1%}" for v in sub["win_rate"]],
            textposition="auto",
            marker_color=color,
        ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="#555", annotation_text="50%")
    fig.update_layout(
        template=_DARK,
        title="Win Rate by Gate",
        yaxis={"title": "Win Rate", "tickformat": ".0%", "range": [0, 1]},
        barmode="group",
        height=420,
        margin={"l": 60, "r": 40, "t": 50, "b": 80},
    )
    return fig


def slippage_scatter_chart(trades: list[TradeRecord]) -> go.Figure:
    """Entry vs fill price scatter, coloured by won/lost."""
    df = m.slippage_stats(trades)
    if df.empty:
        return _empty("No fill data (live trades only)")

    color_map = {True: _WIN, False: _LOSS}
    colors = [color_map.get(w, _PENDING) for w in df["won"]]  # type: ignore[call-overload]
    hover = [
        f"{row['direction'].upper()}<br>"
        f"Entry: {row['entry_price']:.4f}  Fill: {row['fill_price']:.4f}<br>"
        f"Slip: {row['slippage_pct']:.2f}%  Spread: {(row['spread'] or 0)*100:.1f}¢<br>"
        f"PnL: ${row['pnl']:+.2f}"
        for _, row in df.iterrows()
    ]

    lo = min(df["entry_price"].min(), df["fill_price"].min()) * 0.995
    hi = max(df["entry_price"].max(), df["fill_price"].max()) * 1.005

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dot", "color": "#444"}, name="No slippage", showlegend=False))
    fig.add_trace(go.Scatter(
        x=df["entry_price"], y=df["fill_price"],
        mode="markers",
        marker={"color": colors, "size": 7, "opacity": 0.75},
        text=hover,
        hoverinfo="text",
        name="Trades",
    ))
    fig.update_layout(
        template=_DARK,
        title="Entry Price vs Fill Price",
        xaxis_title="Entry (signal price)",
        yaxis_title="Fill (actual price)",
        height=460,
        margin={"l": 60, "r": 40, "t": 50, "b": 60},
    )
    return fig


def streak_profile_chart(trades: list[TradeRecord]) -> go.Figure:
    """Win rate after N consecutive wins/losses."""
    df = m.streak_profile(trades)
    if df.empty:
        return _empty("No streak data (live trades with session tracking only)")

    fig = go.Figure()
    for streak_type, color, label in [
        ("after_wins", "#4fc3f7", "After N wins"),
        ("after_losses", _LOSS, "After N losses"),
    ]:
        sub = df[df["streak_type"] == streak_type]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=label,
            x=sub["streak_len"].astype(str),
            y=sub["win_rate"],
            text=[f"{v:.1%} (n={int(n)})" for v, n in zip(sub["win_rate"], sub["n"])],
            textposition="auto",
            marker_color=color,
        ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="#555", annotation_text="50%")
    fig.update_layout(
        template=_DARK,
        title="Win Rate After Consecutive Wins / Losses",
        xaxis_title="Streak Length",
        yaxis={"title": "Win Rate", "tickformat": ".0%", "range": [0, 1]},
        barmode="group",
        height=420,
    )
    return fig


def pnl_by_hour_chart(trades: list[TradeRecord]) -> go.Figure:
    """Bar chart: PnL and win rate by UTC hour."""
    df = m.pnl_by_hour(trades)
    if df.empty:
        return _empty("No hourly data")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df.index, y=df["total_pnl"], name="Total PnL", marker_color=[_WIN if v >= 0 else _LOSS for v in df["total_pnl"]]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["win_rate"], name="Win Rate", mode="lines+markers", line={"color": _GOLD}),
        secondary_y=True,
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="#555", secondary_y=True)
    fig.update_layout(
        template=_DARK,
        title="PnL & Win Rate by UTC Hour",
        xaxis_title="Hour (UTC)",
        height=420,
    )
    fig.update_yaxes(title_text="PnL ($)", secondary_y=False)
    fig.update_yaxes(title_text="Win Rate", tickformat=".0%", secondary_y=True)
    return fig


# ── util ──────────────────────────────────────────────────────────────────────


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=_DARK,
        annotations=[{"text": msg, "showarrow": False, "font": {"size": 16, "color": "#888"}, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}],
        height=300,
    )
    return fig
