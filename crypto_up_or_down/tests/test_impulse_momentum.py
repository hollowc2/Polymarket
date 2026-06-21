from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from polymarket_algo.core.config import Config
from polymarket_algo.core.types import Strategy
from polymarket_algo.executor.client import Market, PolymarketClient
from polymarket_algo.executor.trader import PaperTrader, Trade, TradingState
from polymarket_algo.strategies.impulse_momentum import ImpulseMomentumStrategy

from scripts.bots.impulse_momentum_bot import (
    book_snapshot,
    entry_price_allowed,
    portfolio_bet_size,
    sell_execution,
    settle_paper_exit,
)


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


def test_sell_execution_walks_bids_and_requires_full_depth() -> None:
    book = {
        "bids": [
            {"price": "0.55", "size": "3"},
            {"price": "0.60", "size": "2"},
        ]
    }
    execution = sell_execution(book, 4)
    assert execution is not None
    assert execution.price == pytest.approx(0.575)
    assert execution.proceeds == pytest.approx(2.30)
    assert sell_execution(book, 6) is None


def test_entry_drift_limit_is_inclusive() -> None:
    assert entry_price_allowed(0.70, 0.73, 0.03)
    assert not entry_price_allowed(0.70, 0.731, 0.03)


def test_crypto_fee_formula_uses_shares() -> None:
    assert PolymarketClient.calculate_fee_amount(100, 0.50, 700) == pytest.approx(1.75)
    assert PolymarketClient.calculate_fee(0.50, 700) == pytest.approx(0.035)


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
        fee_rate_bps=700,
    )
    state = TradingState(trades=[trade], bankroll=100.0)
    settle_paper_exit(state, trade, exit_price=0.60, reason="stop_loss_25%")
    assert trade.settlement_status == "settled"
    assert trade.final_price == pytest.approx(0.60)
    assert trade.fee_amount == pytest.approx(0.1995)
    assert trade.pnl == pytest.approx(-1.1995)
    assert state.bankroll == pytest.approx(98.8005)
    assert state.daily_pnl == pytest.approx(-1.1995)
    assert trade.to_nested_json()["settlement"]["force_exit_reason"] == "stop_loss_25%"


def test_paper_fok_rejects_partial_fill() -> None:
    market = Market(
        timestamp=1_700_000_000,
        slug="btc-updown-5m-1700000000",
        title="BTC Up or Down",
        closed=False,
        outcome=None,
        up_token_id="up",
        down_token_id="down",
        up_price=0.70,
        down_price=0.30,
        volume=1_000,
        accepting_orders=True,
        taker_fee_bps=700,
    )
    trade = PaperTrader().place_bet(
        market,
        "up",
        10.0,
        0.70,
        0,
        precomputed_execution={
            "execution_price": 0.71,
            "fill_pct": 99.0,
            "spread": 0.02,
            "slippage_pct": 0.0,
        },
    )
    assert trade is None


def test_consecutive_loss_pause_resets_next_utc_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "MAX_CONSEC_LOSSES", 5)
    now = datetime.now(UTC)
    losses = [
        Trade(
            timestamp=1_700_000_000 + i,
            market_slug=f"loss-{i}",
            direction="up",
            amount=5.0,
            entry_price=0.5,
            streak_length=0,
            confidence=0.5,
            paper=True,
            won=False,
            settled_at=int(now.timestamp() * 1000),
        )
        for i in range(5)
    ]
    allowed, _ = TradingState(trades=losses).can_trade()
    assert not allowed

    yesterday = int((now - timedelta(days=1)).timestamp() * 1000)
    for trade in losses:
        trade.settled_at = yesterday
    allowed, _ = TradingState(trades=losses).can_trade()
    assert allowed
