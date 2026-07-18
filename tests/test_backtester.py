"""Unit tests: metrics computation + a small end-to-end backtest run."""

from datetime import datetime, timedelta

from backtester import BTTrade, compute_metrics, Backtester
from strategy import Bar


def make_trade(pnl, r, outcome):
    return BTTrade(
        entry_time=datetime(2026, 4, 1), exit_time=datetime(2026, 4, 1, 1),
        direction=1, entry=1.1, exit=1.1, stop_distance=0.0010, lots=0.02,
        gross_pips=0.0, net_pips=0.0, pnl_money=pnl, r_multiple=r,
        outcome=outcome, quality="standard")


def test_metrics_empty():
    m = compute_metrics([], [1000.0], 1000.0)
    assert m["trades"] == 0


def test_metrics_basic_expectancy():
    trades = [make_trade(160, 1.6, "win"), make_trade(160, 1.6, "win"),
              make_trade(-100, -1.0, "loss")]
    m = compute_metrics(trades, [1000, 1160, 1320, 1220], 1000.0)
    assert m["trades"] == 3
    assert m["wins"] == 2 and m["losses"] == 1
    assert abs(m["win_rate"] - 2 / 3) < 1e-3          # win_rate rounded to 4dp
    # expectancy in R: (1.6 + 1.6 - 1.0) / 3 -> avg R
    assert abs(m["expectancy_R"] - (1.6 + 1.6 - 1.0) / 3) < 1e-3
    # profit factor = 320 / 100 = 3.2
    assert abs(float(m["profit_factor"]) - 3.2) < 1e-9


def test_metrics_drawdown_and_streaks():
    trades = [make_trade(-100, -1, "loss"), make_trade(-100, -1, "loss"),
              make_trade(-100, -1, "loss"), make_trade(160, 1.6, "win")]
    curve = [1000, 900, 800, 700, 860]
    m = compute_metrics(trades, curve, 1000.0)
    assert m["longest_loss_streak"] == 3
    assert m["longest_win_streak"] == 1
    # peak 1000 -> trough 700 => max dd = 30%
    assert abs(m["max_drawdown"] - 0.30) < 1e-9


def test_metrics_net_profit():
    trades = [make_trade(50, 0.5, "win"), make_trade(-30, -0.3, "loss")]
    m = compute_metrics(trades, [1000, 1050, 1020], 1000.0)
    assert abs(m["net_profit"] - 20.0) < 1e-9
    assert abs(m["ending_balance"] - 1020.0) < 1e-9


def _flat_bars(n):
    """A long flat series -> strategy should never signal, backtest = 0 trades."""
    bars = []
    t = datetime(2026, 1, 1)
    for _ in range(n):
        bars.append(Bar(time=t, open=1.1, high=1.1001, low=1.0999,
                         close=1.1, volume=1.0))
        t += timedelta(minutes=15)
    return bars


def test_backtest_runs_without_error_on_flat_data():
    cfg = {
        "asia_start": "00:00", "asia_end": "07:00",
        "session_start": "07:00", "session_end": "11:00",
        "hard_flat_time": "20:00",
        "strategy": {"atr_period": 14},
        "risk_per_trade": 0.005, "max_lot": 0.10, "max_open_trades": 1,
        "max_trades_per_day": 2, "max_losses_per_day": 1,
        "max_daily_loss": 0.03, "max_weekly_loss": 0.06,
        "max_losing_days_per_week": 3, "max_drawdown": 0.10,
        "max_spread_pips": 1.2,
        "recovery_enabled": True,
        "recovery_stage_thresholds": {
            "stage1_warning": 0.02, "stage2_defensive": 0.05,
            "stage3_recovery": 0.08, "stage4_shutdown": 0.10},
        "backtest": {"spread_pips": 0.7, "slippage_pips": 0.3,
                     "starting_balance": 1000.0},
    }
    result = Backtester(cfg).run(_flat_bars(3000))
    # flat data -> no compression-gated breakouts -> no trades, no crash
    assert result.metrics["trades"] == 0
    assert result.starting_balance == 1000.0
