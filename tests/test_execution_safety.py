"""Unit tests: the DEMO/LIVE safety latch in the execution engine."""

from execution_engine import ExecutionEngine
from mt5_connector import MT5Connector


def engine(account_mode, live_enabled):
    cfg = {
        "account_mode": account_mode,
        "live_trading_enabled": live_enabled,
        "symbol": "EURUSD", "magic_number": 770524,
        "max_slippage_pips": 0.5,
    }
    return ExecutionEngine(cfg, MT5Connector(cfg))


def test_default_config_is_locked():
    """Shipped defaults (demo + disabled) must NOT unlock live trading."""
    assert engine("demo", False).live_unlocked() is False


def test_demo_mode_never_unlocks_even_if_flag_true():
    assert engine("demo", True).live_unlocked() is False


def test_live_mode_alone_does_not_unlock():
    assert engine("live", False).live_unlocked() is False


def test_both_latches_required_for_live():
    assert engine("live", True).live_unlocked() is True


def test_trading_permitted_refuses_without_connection():
    """No MT5 connection -> trade-mode unreadable -> refuse to trade."""
    permitted, reason = engine("demo", False).trading_permitted()
    assert permitted is False
    assert "unreadable" in reason.lower()
