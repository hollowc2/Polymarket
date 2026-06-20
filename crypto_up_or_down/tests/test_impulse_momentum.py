import pandas as pd
import pytest
from polymarket_algo.core.types import Strategy
from polymarket_algo.executor.trader import Trade, TradingState
from polymarket_algo.strategies.impulse_momentum import ImpulseMomentumStrategy

from scripts.bots.impulse_momentum_bot import book_snapshot, portfolio_bet_size, settle_paper_exit


def evaluate(open_price: float, close_price: float, up_ask: float, down_ask: float):
    frame = pd.DataFrame([{"open": open_price, "close": close_price, "up_ask": up_ask, "down_ask": down_ask}])
    return ImpulseMomentumStrategy().evaluate(frame).iloc[-1]


def test_strategy_conforms_protocol() -> None:
    strategy: Strategy = ImpulseMomentumStrategy()
    result = strategy.evaluate(pd.DataFrame([{"open": 100_000, "close": 100_080, "up_ask": 0.72, "down_ask": 0.30}]))
    assert {"signal", "size", "impulse_usd", "selected_ask"}.issubset(result.columns)


def test_bullish_impulse_requires_matching_skew() -> None:
    assert int(evaluate(100_000, 100_080, 0.72, 0.30)["signal"]) == 1
    assert int(evaluate(100_000, 100_080, 0.60, 0.40)["signal"]) == 0
    assert int(evaluate(100_000, 100_080, 0.72, 0.75)["signal"]) == 0


def test_bearish_impulse_requires_matching_skew() -> None:
    assert int(evaluate(100_000, 99_920, 0.28, 0.73)["signal"]) == -1
    assert int(evaluate(100_000, 99_950, 0.28, 0.73)["signal"]) == 0


def test_larger_than_reference_impulse_still_qualifies() -> None:
    result = evaluate(100_000, 100_150, 0.80, 0.22)
    assert int(result["signal"]) == 1
    assert float(result["impulse_usd"]) == 150.0


def test_book_snapshot_uses_executable_top_levels() -> None:
    snapshot = book_snapshot(
        {
            "bids": [{"price": "0.68", "size": "10"}, {"price": "0.69", "size": "5"}],
            "asks": [{"price": "0.73", "size": "20"}, {"price": "0.72", "size": "50"}],
        }
    )
    assert snapshot is not None
    assert snapshot.best_bid == pytest.approx(0.69)
    assert snapshot.best_ask == pytest.approx(0.72)
    assert snapshot.spread == pytest.approx(0.03)
    assert snapshot.top_ask_notional == pytest.approx(36.0)


def test_book_snapshot_keeps_ask_only_outcome_for_skew_comparison() -> None:
    snapshot = book_snapshot({"bids": [], "asks": [{"price": "0.08", "size": "100"}]})
    assert snapshot is not None
    assert snapshot.best_bid == 0.0
    assert snapshot.best_ask == pytest.approx(0.08)
    assert snapshot.spread == pytest.approx(0.08)


def test_portfolio_sizing_compounds_with_aum() -> None:
    assert portfolio_bet_size(100.0, risk_pct=10.0, max_notional=0.0) == 10.0
    assert portfolio_bet_size(125.0, risk_pct=10.0, max_notional=0.0) == 12.5
    assert portfolio_bet_size(80.0, risk_pct=10.0, max_notional=0.0) == 8.0


def test_portfolio_sizing_honors_optional_dollar_cap() -> None:
    assert portfolio_bet_size(200.0, risk_pct=10.0, max_notional=15.0) == 15.0


def test_paper_exit_realizes_bid_value() -> None:
    trade = Trade(
        timestamp=1_700_000_000,
        market_slug="btc-updown-5m-1700000000",
        direction="up",
        amount=5.0,
        entry_price=0.75,
        execution_price=0.75,
        streak_length=0,
        confidence=0.75,
        paper=True,
        fee_rate_bps=0,
    )
    state = TradingState(trades=[trade], bankroll=100.0)
    settle_paper_exit(state, trade, exit_price=0.60, reason="stop_loss_25%")
    assert trade.settlement_status == "settled"
    assert trade.final_price == pytest.approx(0.60)
    assert trade.pnl == pytest.approx(-1.0)
    assert state.bankroll == pytest.approx(99.0)
    assert state.daily_pnl == pytest.approx(-1.0)
