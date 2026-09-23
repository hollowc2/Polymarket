"""Live mode configuration safety tests."""

from __future__ import annotations

import pytest
from polymarket_algo.core.config import Config
from polymarket_algo.executor.trader import LiveTrader


def _validator() -> LiveTrader:
    return LiveTrader.__new__(LiveTrader)


def test_live_trader_rejects_default_paper_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "APP_MODE", "paper")

    with pytest.raises(ValueError, match="APP_MODE=live"):
        _validator()._validate_live_mode_config()


def test_live_trader_rejects_invalid_app_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "APP_MODE", "prod")

    with pytest.raises(ValueError, match="APP_MODE must be"):
        _validator()._validate_live_mode_config()


def test_live_trader_rejects_missing_wallet_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "APP_MODE", "live")
    monkeypatch.setattr(Config, "SIGNATURE_TYPE", 0)
    monkeypatch.setattr(Config, "WALLET_ADDRESS", "")

    with pytest.raises(ValueError, match="WALLET_ADDRESS required"):
        _validator()._validate_live_mode_config()


def test_live_trader_rejects_mismatched_live_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "APP_MODE", "live")
    monkeypatch.setattr(Config, "SIGNATURE_TYPE", 0)
    monkeypatch.setattr(Config, "WALLET_ADDRESS", "0xabc")
    monkeypatch.setattr(Config, "LIVE_CONFIRM", "YES")

    with pytest.raises(ValueError, match="LIVE_CONFIRM must equal crypto_up_or_down:0xabc"):
        _validator()._validate_live_mode_config()


def test_live_trader_accepts_explicit_wallet_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "APP_MODE", "live")
    monkeypatch.setattr(Config, "SIGNATURE_TYPE", 0)
    monkeypatch.setattr(Config, "WALLET_ADDRESS", "0xAbC")
    monkeypatch.setattr(Config, "LIVE_CONFIRM", "crypto_up_or_down:0xabc")

    _validator()._validate_live_mode_config()


def test_live_trader_uses_funder_for_proxy_wallet_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "APP_MODE", "live")
    monkeypatch.setattr(Config, "SIGNATURE_TYPE", 1)
    monkeypatch.setattr(Config, "FUNDER_ADDRESS", "0xFunder")
    monkeypatch.setattr(Config, "LIVE_CONFIRM", "crypto_up_or_down:0xfunder")

    _validator()._validate_live_mode_config()
