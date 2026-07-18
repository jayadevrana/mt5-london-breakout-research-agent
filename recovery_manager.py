"""
recovery_manager.py
-------------------
The Stage 0-4 capital-protection state machine.

Principle: as drawdown deepens, risk per trade is CUT, never raised.
Recovery happens only through positive expectancy at reduced size.
Risk steps back up ONLY after a proven recovery milestone.
There is NO martingale, NO grid, NO averaging-down anywhere in here.

Stages
  0 Normal       dd < stage1
  1 Warning      stage1 <= dd < stage2   -> reduced risk, A+ only
  2 Defensive    stage2 <= dd < stage3   -> half risk, stronger confirm
  3 Recovery     stage3 <= dd < stage4   -> minimum risk, best setup only
  4 Shutdown     dd >= stage4 (or external trigger) -> ALL trading halts

Pure logic, no MT5 import -> fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass


STAGE_NAMES = {0: "Normal", 1: "Warning", 2: "Defensive",
               3: "Recovery", 4: "Shutdown"}


@dataclass
class RecoveryState:
    stage: int = 0
    risk_per_trade: float = 0.005
    trades_per_day: int = 2
    confirm_closes: int = 1
    both_directions: bool = True
    halted: bool = False
    note: str = "Normal mode"


class RecoveryManager:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.enabled = bool(config.get("recovery_enabled", True))
        th = config.get("recovery_stage_thresholds", {}) or {}
        self.t1 = float(th.get("stage1_warning", 0.02))
        self.t2 = float(th.get("stage2_defensive", 0.05))
        self.t3 = float(th.get("stage3_recovery", 0.08))
        self.t4 = float(th.get("stage4_shutdown", 0.10))
        rk = config.get("recovery_risk_per_stage", {}) or {}
        base = float(config.get("risk_per_trade", 0.005))
        self.risk = {
            0: float(rk.get("stage0", base)),
            1: float(rk.get("stage1", base * 0.70)),
            2: float(rk.get("stage2", base * 0.50)),
            3: float(rk.get("stage3", base * 0.40)),
            4: 0.0,
        }
        # Stage 4 is sticky: once shut down, only a human config flag resumes.
        self._shutdown_latched = False
        self._external_halt_reason = ""

    # -- external (non-drawdown) shutdown triggers -------------------------
    def force_shutdown(self, reason: str) -> None:
        """Called by the engine when a non-drawdown rule is violated, e.g.
        live stats outside the Monte Carlo band, persistent bad spread,
        or a losing streak beyond the MC 95th percentile."""
        self._shutdown_latched = True
        self._external_halt_reason = reason

    def reset_shutdown(self) -> None:
        """A HUMAN action only. Never call this automatically."""
        self._shutdown_latched = False
        self._external_halt_reason = ""

    # -- main evaluation ---------------------------------------------------
    def evaluate(self, drawdown: float) -> RecoveryState:
        """
        drawdown: fraction in [0, 1], the worse of from-initial / from-peak.
        Returns the RecoveryState the rest of the system must obey.
        """
        if not self.enabled:
            return RecoveryState(stage=0, risk_per_trade=self.risk[0],
                                 trades_per_day=2, confirm_closes=1,
                                 both_directions=True, halted=False,
                                 note="Recovery disabled in config")

        if self._shutdown_latched or drawdown >= self.t4:
            if drawdown >= self.t4:
                self._shutdown_latched = True
            reason = self._external_halt_reason or f"drawdown {drawdown:.3f} >= {self.t4:.3f}"
            return RecoveryState(stage=4, risk_per_trade=0.0, trades_per_day=0,
                                 confirm_closes=99, both_directions=False,
                                 halted=True,
                                 note=f"SHUTDOWN -- {reason}. Human review required.")

        if drawdown >= self.t3:
            return RecoveryState(stage=3, risk_per_trade=self.risk[3],
                                 trades_per_day=1, confirm_closes=2,
                                 both_directions=False,
                                 note="Recovery mode: best setup only, minimum risk.")

        if drawdown >= self.t2:
            return RecoveryState(stage=2, risk_per_trade=self.risk[2],
                                 trades_per_day=1, confirm_closes=2,
                                 both_directions=True,
                                 note="Defensive mode: half risk, stronger confirmation.")

        if drawdown >= self.t1:
            return RecoveryState(stage=1, risk_per_trade=self.risk[1],
                                 trades_per_day=1, confirm_closes=1,
                                 both_directions=False,
                                 note="Warning mode: A+ setups only, reduced risk.")

        return RecoveryState(stage=0, risk_per_trade=self.risk[0],
                             trades_per_day=2, confirm_closes=1,
                             both_directions=True, note="Normal mode.")

    # -- recovery milestone (controls when risk may step back UP) ----------
    def may_step_up(self, profitable_days_in_row: int,
                    drawdown_recovered_fraction: float) -> bool:
        """
        Risk may move one stage toward normal ONLY when a milestone is met:
          * 3 consecutive profitable days, OR
          * 25% of the peak drawdown has been recovered.
        This is the ONLY upward path. Never raise risk after a single win.
        """
        return profitable_days_in_row >= 3 or drawdown_recovered_fraction >= 0.25

    # -- recovery analytics (the questions Part 5 must answer) -------------
    @staticmethod
    def expected_trades_to_recover(drawdown_in_R: float,
                                   expectancy_R: float) -> float:
        """trades ~= drawdown_in_R / expectancy. Negative/zero edge -> inf."""
        if expectancy_R <= 0:
            return float("inf")
        return drawdown_in_R / expectancy_R

    @staticmethod
    def risk_of_ruin(edge_fraction: float, units_to_ruin: float) -> float:
        """
        RoR ~= ((1 - A) / (1 + A)) ** u   for a positive edge A.
        edge_fraction A <= 0  -> ruin is effectively certain -> returns 1.0.
        """
        if edge_fraction <= 0:
            return 1.0
        a = min(edge_fraction, 0.999)
        base = (1.0 - a) / (1.0 + a)
        try:
            return max(0.0, min(1.0, base ** units_to_ruin))
        except OverflowError:
            return 0.0
