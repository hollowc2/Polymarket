"""deepanal — Strategy Explorer

Launch with:
    uv run python scripts/run_deepanal.py
    # or directly:
    uv run streamlit run scripts/deepanal/app.py
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Make `deepanal` importable whether Streamlit runs this file directly or via
# the run_deepanal.py launcher.
_scripts_dir = Path(__file__).parents[1]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import pandas as pd
import streamlit as st

from deepanal import charts, metrics
from deepanal.aligner import align, load_ohlcv
from deepanal.loader import STATE_DIR, discover_bots, from_backtest_result, load_bot_trades
from deepanal.models import TradeRecord

ROOT = Path(__file__).parents[2]

# ── strategy registry for backtest runner ─────────────────────────────────────
# Maps display name → (module_path, class_name)
STRATEGY_REGISTRY: dict[str, tuple[str, str]] = {
    "StreakReversal":        ("polymarket_algo.strategies.streak_reversal",   "StreakReversalStrategy"),
    "StreakADX":             ("polymarket_algo.strategies.streak_adx",        "StreakADXStrategy"),
    "StreakRSI":             ("polymarket_algo.strategies.streak_rsi",        "StreakRSIStrategy"),
    "CandleDirection":       ("polymarket_algo.strategies.candle_direction",  "CandleDirectionStrategy"),
    "ThreeBarMoMo":          ("polymarket_algo.strategies.three_bar_momo",    "ThreeBarMoMoStrategy"),
    "ApexStrategy":          ("polymarket_algo.strategies.apex_strategy",     "ApexStrategy"),
    "BollingerSqueeze":      ("polymarket_algo.strategies.bollinger_squeeze", "BollingerSqueezeStrategy"),
    "PinBarReversal":        ("polymarket_algo.strategies.pin_bar",           "PinBarReversalStrategy"),
    "CVDDivergence":         ("polymarket_algo.strategies.cvd_divergence",    "CVDDivergenceStrategy"),
    "DeltaFlip":             ("polymarket_algo.strategies.delta_flip",        "DeltaFlipStrategy"),
    "FundingRateExtremes":   ("polymarket_algo.strategies.funding_rate_extremes", "FundingRateExtremesStrategy"),
    "LiquidationCascade":    ("polymarket_algo.strategies.liquidation_cascade", "LiquidationCascadeStrategy"),
    "OIRateOfChange":        ("polymarket_algo.strategies.oi_roc",            "OIRateOfChangeStrategy"),
    "SpotPerpBasis":         ("polymarket_algo.strategies.spot_perp_basis",   "SpotPerpBasisStrategy"),
    "CoinbasePremium":       ("polymarket_algo.strategies.coinbase_premium",  "CoinbasePremiumStrategy"),
    "RealizedVolRegime":     ("polymarket_algo.strategies.rv_regime",         "RealizedVolRegimeStrategy"),
    "HLOrderFlowReversal":   ("polymarket_algo.strategies.hl_orderflow_reversal",  "HLOrderFlowReversalStrategy"),
    "HLOrderFlowMomentum":   ("polymarket_algo.strategies.hl_orderflow_momentum",  "HLOrderFlowMomentumStrategy"),
    "CorrBreak":             ("polymarket_algo.strategies.corr_break",        "CorrBreakStrategy"),
}

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="deepanal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card { background:#1e1e2e; border-radius:8px; padding:12px 16px; margin:4px 0; }
    .metric-value { font-size:1.6rem; font-weight:700; }
    .metric-label { font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:.05em; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading OHLCV…")
def _load_ohlcv(asset: str, tf: str, kind: str) -> pd.DataFrame:
    return load_ohlcv(asset, tf, kind)


@st.cache_data(show_spinner="Loading trade history…")
def _load_bot(bot_name: str) -> list[TradeRecord]:
    return load_bot_trades(bot_name)


def _run_backtest(strategy_name: str, ohlcv: pd.DataFrame) -> list[TradeRecord] | None:
    """Instantiate strategy, run backtest, return TradeRecord list."""
    from polymarket_algo.backtest.engine import run_backtest

    mod_path, cls_name = STRATEGY_REGISTRY[strategy_name]
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        strategy = cls()
        result = run_backtest(ohlcv, strategy)
        return from_backtest_result(result, strategy_name)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        return None


def _kpi_row(s: dict) -> None:
    """Render 5 KPI cards in a single row."""
    cols = st.columns(5)
    items = [
        ("Trades", f"{s['trade_count']:,}"),
        ("Win Rate", f"{s['win_rate']:.1%}"),
        ("Total PnL", f"${s['total_pnl']:+.2f}"),
        ("Sharpe", f"{s['sharpe']:.2f}"),
        ("Max DD", f"${s['max_drawdown']:.2f}"),
    ]
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("deepanal")
    st.caption("Strategy Explorer")
    st.divider()

    source = st.radio("Data source", ["Live / Paper", "Backtest"], horizontal=True)

    st.subheader("OHLCV")
    col_a, col_b = st.columns(2)
    asset = col_a.selectbox("Asset", ["btc"], index=0)
    tf = col_b.selectbox("TF", ["5m"], index=0)
    kind = st.selectbox("Kind", ["perp", "spot"], index=0)

    trades: list[TradeRecord] = []
    ohlcv_df: pd.DataFrame | None = None

    # Load OHLCV (shared by both sources)
    try:
        ohlcv_df = _load_ohlcv(asset, tf, kind)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    ohlcv_range = (ohlcv_df.index.min().date(), ohlcv_df.index.max().date())

    st.divider()

    if source == "Live / Paper":
        st.subheader("Live trades")
        st.caption(f"State dir: `{STATE_DIR}`")

        available_bots = discover_bots()
        if not available_bots:
            st.warning(f"No history files found in `{STATE_DIR}`.")
        else:
            bot_sel = st.selectbox("Bot", available_bots)
            min_amt = st.slider("Min trade amount ($)", 0.0, 50.0, 0.0, 0.5)

            all_trades = _load_bot(bot_sel)

            if all_trades:
                date_min = min(t.open_time.date() for t in all_trades)
                date_max = max(t.open_time.date() for t in all_trades)
                date_range = st.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    d0 = datetime(date_range[0].year, date_range[0].month, date_range[0].day, tzinfo=UTC)
                    d1 = datetime(date_range[1].year, date_range[1].month, date_range[1].day, 23, 59, 59, tzinfo=UTC)
                    all_trades = [t for t in all_trades if d0 <= t.open_time <= d1]

                trades = [t for t in all_trades if t.amount >= min_amt]
                st.caption(f"{len(trades)} trades")

    else:  # Backtest
        st.subheader("Backtest")
        strategy_name = st.selectbox("Strategy", list(STRATEGY_REGISTRY.keys()))

        # Date slice for OHLCV
        date_range_bt = st.date_input(
            "OHLCV range",
            value=(ohlcv_range[1] - timedelta(days=180), ohlcv_range[1]),
            min_value=ohlcv_range[0],
            max_value=ohlcv_range[1],
        )

        if st.button("Run backtest", type="primary"):
            if isinstance(date_range_bt, (list, tuple)) and len(date_range_bt) == 2:
                d0 = pd.Timestamp(date_range_bt[0], tz="UTC")
                d1 = pd.Timestamp(date_range_bt[1], tz="UTC")
                ohlcv_slice = ohlcv_df.loc[d0:d1]
            else:
                ohlcv_slice = ohlcv_df

            with st.spinner(f"Running {strategy_name}…"):
                result = _run_backtest(strategy_name, ohlcv_slice)

            if result is not None:
                st.session_state["bt_trades"] = result
                st.session_state["bt_ohlcv"] = ohlcv_slice
                st.success(f"Done — {len(result)} trades")

        if "bt_trades" in st.session_state:
            trades = st.session_state["bt_trades"]
            ohlcv_df = st.session_state["bt_ohlcv"]
            st.caption(f"{len(trades)} backtest trades")

    st.divider()
    show_vol = st.toggle("Volume subplot", value=True)
    show_eq = st.toggle("Equity subplot", value=True)


# ── main area ─────────────────────────────────────────────────────────────────

st.title("Strategy Explorer")

if not trades:
    if source == "Live / Paper":
        st.info("No trades loaded. Check the history file path or strategy filter in the sidebar.")
    else:
        st.info("Select a strategy and click **Run backtest** in the sidebar.")
    st.stop()

# Summary KPIs
s = metrics.summary(trades)
_kpi_row(s)
st.divider()

# Align trades to OHLCV
ohlcv_window, trades_df = align(trades, ohlcv_df)

# Tabs
tab_chart, tab_time, tab_fill, tab_gate, tab_streak, tab_raw = st.tabs([
    "Chart", "Time Analysis", "Fill Quality", "Gate Attribution", "Streak Profile", "Raw Trades",
])

with tab_chart:
    fig = charts.candlestick_trades(ohlcv_window, trades, show_volume=show_vol, show_equity=show_eq)
    st.plotly_chart(fig, width="stretch")

with tab_time:
    col1, col2 = st.columns(2)
    metric_sel = col1.radio("Metric", ["win_rate", "pnl"], horizontal=True)
    with col1:
        st.plotly_chart(charts.heatmap_hour_weekday(trades, metric_sel), width="stretch")
    with col2:
        st.plotly_chart(charts.pnl_by_hour_chart(trades), width="stretch")
    st.plotly_chart(charts.equity_curve_chart(trades), width="stretch")

with tab_fill:
    if any(t.source == "live" for t in trades):
        st.plotly_chart(charts.slippage_scatter_chart(trades), width="stretch")
        slip_df = metrics.slippage_stats(trades)
        if not slip_df.empty:
            st.subheader("Slippage summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg slippage", f"{slip_df['slippage_pct'].mean():.3f}%")
            col2.metric("Max slippage", f"{slip_df['slippage_pct'].max():.3f}%")
            avg_spread = slip_df["spread"].dropna()
            col3.metric("Avg spread", f"{avg_spread.mean()*100:.1f}¢" if not avg_spread.empty else "N/A")
            st.dataframe(
                slip_df[["open_time", "direction", "entry_price", "fill_price", "slippage_pct", "spread", "overpay_vs_ask", "won", "pnl"]]
                .sort_values("open_time", ascending=False)
                .reset_index(drop=True),
                width="stretch",
            )
    else:
        st.info("Fill quality data is only available for live/paper trades.")

with tab_gate:
    if any(t.source == "live" and t.gate_name for t in trades):
        st.plotly_chart(charts.gate_breakdown_chart(trades), width="stretch")
        gate_df = metrics.pnl_by_gate(trades)
        if not gate_df.empty:
            st.dataframe(gate_df, width="stretch")
    else:
        st.info("Gate attribution is only available for live/paper trades with active gates.")

with tab_streak:
    st.plotly_chart(charts.streak_profile_chart(trades), width="stretch")
    streak_df = metrics.streak_profile(trades)
    if not streak_df.empty:
        st.dataframe(streak_df, width="stretch")

with tab_raw:
    display_cols = [
        "id", "strategy", "source", "open_time", "direction",
        "amount", "entry_price", "fill_price", "won", "pnl",
        "gate_name", "gate_boosted", "slippage_pct", "hour_utc", "day_of_week",
        "consecutive_wins", "consecutive_losses",
    ]
    display_df = trades_df[[c for c in display_cols if c in trades_df.columns]].copy()
    if "open_time" in display_df.columns:
        display_df = display_df.sort_values("open_time", ascending=False)

    # Colour won column
    st.dataframe(
        display_df.reset_index(drop=True),
        width="stretch",
        column_config={
            "won": st.column_config.CheckboxColumn("Won"),
            "pnl": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
            "entry_price": st.column_config.NumberColumn("Entry", format="%.4f"),
            "fill_price": st.column_config.NumberColumn("Fill", format="%.4f"),
            "slippage_pct": st.column_config.NumberColumn("Slip %", format="%.2f%%"),
        },
    )
    if st.button("Export CSV"):
        csv = display_df.to_csv(index=False)
        st.download_button("Download trades.csv", csv, "trades.csv", "text/csv")
