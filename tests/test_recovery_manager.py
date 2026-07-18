"""Unit tests: recovery-mode stage transitions and recovery analytics."""

from recovery_manager import RecoveryManager


def cfg():
    return {
        "recovery_enabled": True,
        "risk_per_trade": 0.005,
        "recovery_stage_thresholds": {
            "stage1_warning": 0.02, "stage2_defensive": 0.05,
            "stage3_recovery": 0.08, "stage4_shutdown": 0.10,
        },
        "recovery_risk_per_stage": {
            "stage0": 0.005, "stage1": 0.0035, "stage2": 0.0025,
            "stage3": 0.0020, "stage4": 0.0,
        },
    }


def test_stage0_normal():
    rm = RecoveryManager(cfg())
    s = rm.evaluate(0.01)
    assert s.stage == 0 and s.risk_per_trade == 0.005 and not s.halted


def test_stage1_warning():
    s = RecoveryManager(cfg()).evaluate(0.03)
    assert s.stage == 1 and s.risk_per_trade == 0.0035


def test_stage2_defensive():
    s = RecoveryManager(cfg()).evaluate(0.06)
    assert s.stage == 2 and s.risk_per_trade == 0.0025 and s.confirm_closes == 2


def test_stage3_recovery():
    s = RecoveryManager(cfg()).evaluate(0.09)
    assert s.stage == 3 and s.risk_per_trade == 0.0020
    assert s.both_directions is False and s.trades_per_day == 1


def test_stage4_shutdown_halts():
    s = RecoveryManager(cfg()).evaluate(0.11)
    assert s.stage == 4 and s.halted is True and s.risk_per_trade == 0.0


def test_stage4_is_sticky():
    """Once shut down, recovering equity must NOT auto-resume trading."""
    rm = RecoveryManager(cfg())
    rm.evaluate(0.12)                 # triggers shutdown
    s = rm.evaluate(0.01)             # equity 'recovered'
    assert s.halted is True           # still latched
    rm.reset_shutdown()               # only a human reset clears it
    assert rm.evaluate(0.01).halted is False


def test_external_force_shutdown():
    rm = RecoveryManager(cfg())
    rm.force_shutdown("spread persistently above test assumptions")
    s = rm.evaluate(0.0)
    assert s.halted is True and s.stage == 4


def test_risk_only_steps_up_on_milestone():
    rm = RecoveryManager(cfg())
    assert rm.may_step_up(profitable_days_in_row=1,
                          drawdown_recovered_fraction=0.0) is False
    assert rm.may_step_up(profitable_days_in_row=3,
                          drawdown_recovered_fraction=0.0) is True
    assert rm.may_step_up(profitable_days_in_row=0,
                          drawdown_recovered_fraction=0.25) is True


def test_risk_never_increases_with_drawdown():
    """Core anti-martingale property: deeper drawdown => lower risk."""
    rm = RecoveryManager(cfg())
    risks = [rm.evaluate(dd).risk_per_trade
             for dd in (0.0, 0.03, 0.06, 0.09)]
    assert risks == sorted(risks, reverse=True)


def test_expected_trades_to_recover():
    rm = RecoveryManager(cfg())
    assert rm.expected_trades_to_recover(40.0, 0.12) == 40.0 / 0.12
    # zero/negative edge => recovery not expected
    assert rm.expected_trades_to_recover(40.0, 0.0) == float("inf")


def test_risk_of_ruin_certain_without_edge():
    rm = RecoveryManager(cfg())
    assert rm.risk_of_ruin(edge_fraction=0.0, units_to_ruin=20) == 1.0
    assert rm.risk_of_ruin(edge_fraction=-0.1, units_to_ruin=20) == 1.0
    # positive edge => small RoR
    assert rm.risk_of_ruin(edge_fraction=0.1, units_to_ruin=20) < 0.2
