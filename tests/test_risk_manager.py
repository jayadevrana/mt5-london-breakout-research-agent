"""Unit tests: position sizing, drawdown, trade-permission gate, SL validation."""

import math

from risk_manager import RiskManager, AccountState, SymbolSpec, round_to_step


def base_config():
    return {
        "risk_per_trade": 0.005, "max_lot": 0.10, "max_open_trades": 1,
        "max_trades_per_day": 2, "max_losses_per_day": 1,
        "max_daily_loss": 0.03, "max_weekly_loss": 0.06,
        "max_losing_days_per_week": 3, "max_drawdown": 0.10,
        "max_spread_pips": 1.2,
    }


def eurusd():
    return SymbolSpec(pip_size=0.0001, pip_value_per_lot=10.0,
                      volume_min=0.01, volume_max=100.0, volume_step=0.01)


# -- position sizing -------------------------------------------------------
def test_position_size_basic_risk():
    rm = RiskManager(base_config())
    # equity 1000, risk 0.5% = $5; stop 20 pips => loss/lot = 20*10 = $200
    # lots = 5/200 = 0.025 -> rounds down to 0.02
    lots = rm.position_size(1000.0, 0.0020, eurusd())
    assert lots == 0.02


def test_position_size_clamped_to_max_lot():
    rm = RiskManager(base_config())
    # huge equity, tiny stop -> would size enormous; must clamp to max_lot 0.10
    lots = rm.position_size(1_000_000.0, 0.0001, eurusd())
    assert lots <= 0.10


def test_position_size_zero_when_too_small():
    rm = RiskManager(base_config())
    # tiny equity + huge stop -> cannot meet volume_min within risk cap
    lots = rm.position_size(50.0, 0.0500, eurusd())
    assert lots == 0.0


def test_position_size_rejects_bad_inputs():
    rm = RiskManager(base_config())
    assert rm.position_size(1000.0, 0.0, eurusd()) == 0.0
    assert rm.position_size(0.0, 0.0020, eurusd()) == 0.0
    assert rm.position_size(1000.0, 0.0020, eurusd(), risk_fraction=0.0) == 0.0


def test_round_to_step():
    assert round_to_step(0.0279, 0.01) == 0.02
    assert round_to_step(0.10, 0.01) == 0.10
    assert round_to_step(0.005, 0.01) == 0.0


# -- drawdown --------------------------------------------------------------
def test_drawdown_uses_worse_of_two():
    rm = RiskManager(base_config())
    acc = AccountState(equity=900, balance=900, start_of_day_equity=1000,
                       start_of_week_equity=1000, initial_capital=1000,
                       equity_peak=1100)
    # from initial: 10%; from peak: ~18.2% -> worse is from peak
    dd = rm.current_drawdown(acc)
    assert math.isclose(dd, (1100 - 900) / 1100, rel_tol=1e-9)


def test_drawdown_never_negative():
    rm = RiskManager(base_config())
    acc = AccountState(equity=1200, balance=1200, start_of_day_equity=1000,
                       start_of_week_equity=1000, initial_capital=1000,
                       equity_peak=1200)
    assert rm.current_drawdown(acc) == 0.0


# -- trade permission gate -------------------------------------------------
def healthy_state(**over):
    s = AccountState(equity=1000, balance=1000, start_of_day_equity=1000,
                     start_of_week_equity=1000, initial_capital=1000,
                     equity_peak=1000, open_trades=0, trades_today=0,
                     losses_today=0, losing_days_this_week=0,
                     spread_pips=0.5, connected=True, symbol_info_ok=True,
                     history_ok=True, in_news_blackout=False, in_session=True)
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_can_trade_allows_healthy():
    rm = RiskManager(base_config())
    assert rm.can_trade(healthy_state()).allowed is True


def test_blocks_on_disconnect():
    rm = RiskManager(base_config())
    r = rm.can_trade(healthy_state(connected=False))
    assert not r.allowed and "mt5_disconnected" in r.failed_gates


def test_blocks_on_wide_spread():
    rm = RiskManager(base_config())
    r = rm.can_trade(healthy_state(spread_pips=2.0))
    assert not r.allowed
    assert any("spread" in g for g in r.failed_gates)


def test_blocks_on_max_drawdown():
    rm = RiskManager(base_config())
    r = rm.can_trade(healthy_state(equity=890, equity_peak=1000))
    assert not r.allowed
    assert any("max_drawdown" in g for g in r.failed_gates)


def test_blocks_on_daily_loss_limit():
    rm = RiskManager(base_config())
    r = rm.can_trade(healthy_state(equity=965))  # -3.5% on the day
    assert not r.allowed
    assert any("daily_loss" in g for g in r.failed_gates)


def test_blocks_on_news_and_session():
    rm = RiskManager(base_config())
    assert not rm.can_trade(healthy_state(in_news_blackout=True)).allowed
    assert not rm.can_trade(healthy_state(in_session=False)).allowed


def test_blocks_when_loss_count_hit():
    rm = RiskManager(base_config())
    r = rm.can_trade(healthy_state(losses_today=1))
    assert not r.allowed and "max_losses_per_day" in r.failed_gates
