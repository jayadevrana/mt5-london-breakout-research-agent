"""
risk_manager.py
---------------
Position sizing + the trade-permission gate. Pure logic, no MT5 import,
so it is fully unit-testable.

Core ideas
  * Size from EQUITY, never balance.
  * Risk a fixed fraction per trade; clamp to broker + config lot limits.
  * A trade is permitted ONLY if every gate passes (fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AccountState:
    """A snapshot of everything the risk manager needs to decide."""
    equity: float
    balance: float
    start_of_day_equity: float
    start_of_week_equity: float
    initial_capital: float
    equity_peak: float
    open_trades: int = 0
    trades_today: int = 0
    losses_today: int = 0
    losing_days_this_week: int = 0
    spread_pips: float = 0.0
    connected: bool = True
    symbol_info_ok: bool = True
    history_ok: bool = True
    in_news_blackout: bool = False
    in_session: bool = True


@dataclass
class SymbolSpec:
    """Broker-reported symbol constraints."""
    pip_size: float = 0.0001          # 0.0001 for EURUSD, 0.01 for JPY pairs
    pip_value_per_lot: float = 10.0   # account-currency value of 1 pip per 1.0 lot
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01


@dataclass
class PermissionResult:
    allowed: bool
    reason: str
    failed_gates: list = field(default_factory=list)


def round_to_step(value: float, step: float) -> float:
    """Round a lot size DOWN to a valid broker volume step."""
    if step <= 0:
        return value
    n = int(value / step + 1e-9)
    return round(n * step, 8)


class RiskManager:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        rk = config.get("risk_per_trade", 0.005)
        self.base_risk = float(rk)
        self.max_lot = float(config.get("max_lot", 0.10))
        self.max_open_trades = int(config.get("max_open_trades", 1))
        self.max_trades_per_day = int(config.get("max_trades_per_day", 2))
        self.max_losses_per_day = int(config.get("max_losses_per_day", 1))
        self.max_daily_loss = float(config.get("max_daily_loss", 0.03))
        self.max_weekly_loss = float(config.get("max_weekly_loss", 0.06))
        self.max_losing_days = int(config.get("max_losing_days_per_week", 3))
        self.max_drawdown = float(config.get("max_drawdown", 0.10))
        self.max_spread = float(config.get("max_spread_pips", 1.2))

    # -- position sizing ---------------------------------------------------
    def position_size(
        self,
        equity: float,
        stop_distance_price: float,
        symbol: SymbolSpec,
        risk_fraction: Optional[float] = None,
    ) -> float:
        """
        Lots such that a full stop-loss hit loses `risk_fraction` of equity.

        risk_money   = equity * risk_fraction
        stop_pips    = stop_distance_price / pip_size
        loss_per_lot = stop_pips * pip_value_per_lot
        lots         = risk_money / loss_per_lot
        Result is clamped to [volume_min, min(volume_max, max_lot)] and
        rounded DOWN to volume_step. Returns 0.0 if no valid size exists.
        """
        rf = self.base_risk if risk_fraction is None else float(risk_fraction)
        if rf <= 0 or equity <= 0 or stop_distance_price <= 0:
            return 0.0
        if symbol.pip_size <= 0 or symbol.pip_value_per_lot <= 0:
            return 0.0

        risk_money = equity * rf
        stop_pips = stop_distance_price / symbol.pip_size
        loss_per_lot = stop_pips * symbol.pip_value_per_lot
        if loss_per_lot <= 0:
            return 0.0

        lots = risk_money / loss_per_lot
        ceiling = min(symbol.volume_max, self.max_lot)
        lots = min(lots, ceiling)
        lots = round_to_step(lots, symbol.volume_step)

        if lots < symbol.volume_min:
            # Cannot place a trade small enough to respect the risk cap.
            return 0.0
        return lots

    # -- trade permission gate --------------------------------------------
    def can_trade(self, acc: AccountState) -> PermissionResult:
        """Fail-closed: every gate must pass. Order = most critical first."""
        failed: list = []

        # --- hard infrastructure gates (never bypass) --------------------
        if not acc.connected:
            failed.append("mt5_disconnected")
        if not acc.symbol_info_ok:
            failed.append("symbol_info_unreadable")
        if not acc.history_ok:
            failed.append("history_missing")
        if acc.equity <= 0:
            failed.append("equity_unreadable")

        # --- drawdown / shutdown -----------------------------------------
        dd = self.current_drawdown(acc)
        if dd >= self.max_drawdown:
            failed.append(f"max_drawdown_breached({dd:.3f})")

        # --- daily / weekly loss limits ----------------------------------
        if acc.start_of_day_equity > 0:
            day_loss = (acc.start_of_day_equity - acc.equity) / acc.start_of_day_equity
            if day_loss >= self.max_daily_loss:
                failed.append(f"daily_loss_limit({day_loss:.3f})")
        if acc.start_of_week_equity > 0:
            wk_loss = (acc.start_of_week_equity - acc.equity) / acc.start_of_week_equity
            if wk_loss >= self.max_weekly_loss:
                failed.append(f"weekly_loss_limit({wk_loss:.3f})")
        if acc.losing_days_this_week >= self.max_losing_days:
            failed.append("max_losing_days_reached")

        # --- count / streak limits ---------------------------------------
        if acc.open_trades >= self.max_open_trades:
            failed.append("max_open_trades")
        if acc.trades_today >= self.max_trades_per_day:
            failed.append("max_trades_per_day")
        if acc.losses_today >= self.max_losses_per_day:
            failed.append("max_losses_per_day")

        # --- market-condition gates --------------------------------------
        if acc.spread_pips > self.max_spread:
            failed.append(f"spread_too_wide({acc.spread_pips:.2f})")
        if acc.in_news_blackout:
            failed.append("news_blackout")
        if not acc.in_session:
            failed.append("outside_session")

        if failed:
            return PermissionResult(False, "; ".join(failed), failed)
        return PermissionResult(True, "all gates passed", [])

    # -- drawdown ----------------------------------------------------------
    def current_drawdown(self, acc: AccountState) -> float:
        """Worse (larger) of drawdown-from-initial and drawdown-from-peak."""
        dd_initial = 0.0
        if acc.initial_capital > 0:
            dd_initial = (acc.initial_capital - acc.equity) / acc.initial_capital
        dd_peak = 0.0
        if acc.equity_peak > 0:
            dd_peak = (acc.equity_peak - acc.equity) / acc.equity_peak
        return max(0.0, dd_initial, dd_peak)
