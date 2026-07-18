"""
main.py
-------
Orchestrator for mt5_quant_trading_agent.

Usage
  python main.py --check     run preflight only -- NO trading, just checks
  python main.py             run the live loop (DEMO unless config unlocked)

Safety
  * Default config = demo, live_trading_enabled = false.
  * The execution engine refuses orders on a REAL account unless BOTH
    config latches are open.
  * Every loop iteration re-checks the full trade-permission gate.
  * Kill switch: drawdown / shutdown -> flatten everything and stop.

This is a research-preview demo agent. Validate in the MT5 Strategy Tester
and trade demo for months before considering anything else.
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from datetime import datetime, timezone

from backtester import load_config
from mt5_connector import MT5Connector, MT5_AVAILABLE
from strategy import Strategy
from risk_manager import RiskManager, AccountState
from recovery_manager import RecoveryManager
from execution_engine import ExecutionEngine
from trade_logger import logger
from daily_review import DayStats, end_of_day


# --------------------------------------------------------------------------
def parse_hhmm(s: str):
    h, m = s.split(":")
    return h, m


def load_news_calendar(path: str):
    """Load high-impact events. CSV columns: datetime,impact (optional)."""
    import csv
    import os
    events = []
    if not path or not os.path.exists(path):
        return None  # signals 'calendar missing'
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("datetime") or row.get("time") or "").strip()
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    events.append(datetime.strptime(raw, fmt))
                    break
                except ValueError:
                    continue
    return events


def in_news_blackout(events, now, before_min, after_min) -> bool:
    if not events:
        return False
    for e in events:
        delta = (now - e).total_seconds() / 60.0
        if -before_min <= delta <= after_min:
            return True
    return False


# --------------------------------------------------------------------------
def preflight(config: dict) -> bool:
    """Connection / account / symbol / history checks. No trading."""
    logger.info("=== PREFLIGHT START ===")
    ok = True

    if not MT5_AVAILABLE:
        logger.error("MetaTrader5 package not installed/available.")
        logger.info("Install MT5 terminal + 'pip install MetaTrader5' (Windows).")
        return False

    conn = MT5Connector(config)
    if not conn.connect():
        logger.error("PREFLIGHT FAIL: cannot connect to MT5 terminal.")
        return False

    acc = conn.account()
    if acc is None:
        logger.error("PREFLIGHT FAIL: account info unreadable.")
        ok = False
    else:
        logger.info("account ok", equity=acc["equity"], currency=acc["currency"])

    spec = conn.symbol_spec()
    if spec is None:
        logger.error("PREFLIGHT FAIL: symbol info unreadable.")
        ok = False
    else:
        logger.info("symbol ok", pip_size=spec.pip_size,
                    pip_value=round(spec.pip_value_per_lot, 4))

    bars = conn.get_bars(500)
    if not bars or len(bars) < 200:
        logger.error("PREFLIGHT FAIL: insufficient history.")
        ok = False
    else:
        logger.info("history ok", bars=len(bars))

    spread = conn.spread_pips()
    if spread is None:
        logger.error("PREFLIGHT FAIL: spread unreadable.")
        ok = False
    else:
        logger.info("spread ok", spread_pips=round(spread, 2))

    eng = ExecutionEngine(config, conn)
    permitted, why = eng.trading_permitted()
    logger.info("trade-mode check", permitted=permitted, detail=why)

    conn.shutdown()
    logger.info(f"=== PREFLIGHT {'PASS' if ok else 'FAIL'} ===")
    return ok


# --------------------------------------------------------------------------
class TradingAgent:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.conn = MT5Connector(config)
        self.strategy = Strategy(config)
        self.risk = RiskManager(config)
        self.recovery = RecoveryManager(config)
        self.exec = ExecutionEngine(config, self.conn)
        self.poll = int(config.get("engine", {}).get("poll_seconds", 15))
        self.initial_capital = float(config.get("engine", {})
                                     .get("initial_capital", 1000.0))
        self.magic = int(config.get("magic_number", 770524))
        h, m = parse_hhmm(config.get("hard_flat_time", "20:00"))
        self.hard_flat = (int(h), int(m))
        news = config.get("news", {}) or {}
        self.news_enabled = bool(news.get("enabled", True))
        self.news_events = load_news_calendar(news.get("calendar_file", ""))
        self.news_before = int(news.get("blackout_minutes_before", 15))
        self.news_after = int(news.get("blackout_minutes_after", 15))
        # runtime state
        self.equity_peak = self.initial_capital
        self.cur_day = None
        self.cur_week = None
        self.trades_today = 0
        self.losses_today = 0
        self.dirs_today: set = set()
        self.losing_days_week = 0
        self.day_start_equity = self.initial_capital
        self.week_start_equity = self.initial_capital
        self.last_bar_time = None
        self.known_tickets: dict = {}  # ticket -> last seen profit
        self.day_stats: DayStats | None = None
        self.day_max_dd = 0.0

    # -- helpers -----------------------------------------------------------
    def _rollover(self, now: datetime, equity: float) -> None:
        d = now.date()
        wk = now.isocalendar()[1]
        if d != self.cur_day:
            if self.cur_day is not None:
                if self.losses_today > 0:
                    self.losing_days_week += 1
                self._finish_day(equity)
            self.cur_day = d
            self.trades_today = 0
            self.losses_today = 0
            self.dirs_today = set()
            self.day_start_equity = equity
            self.day_max_dd = 0.0
            self.day_stats = DayStats(
                date=str(d), start_balance=equity, end_balance=equity,
                start_equity=equity, end_equity=equity)
        if wk != self.cur_week:
            self.cur_week = wk
            self.losing_days_week = 0
            self.week_start_equity = equity

    def _finish_day(self, equity: float) -> None:
        if self.day_stats is None:
            return
        self.day_stats.end_balance = equity
        self.day_stats.end_equity = equity
        self.day_stats.trades = self.trades_today
        self.day_stats.losses = self.losses_today
        self.day_stats.wins = max(0, self.trades_today - self.losses_today)
        self.day_stats.net_pl = equity - self.day_start_equity
        self.day_stats.max_intraday_dd = self.day_max_dd
        self.day_stats.recovery_stage = self.recovery.evaluate(
            self._drawdown(equity)).stage
        try:
            end_of_day(self.day_stats)
        except Exception as e:  # noqa: BLE001
            logger.error("daily review failed", err=str(e))

    def _drawdown(self, equity: float) -> float:
        dd_i = ((self.initial_capital - equity) / self.initial_capital
                if self.initial_capital > 0 else 0.0)
        dd_p = ((self.equity_peak - equity) / self.equity_peak
                if self.equity_peak > 0 else 0.0)
        return max(0.0, dd_i, dd_p)

    def _track_closed_positions(self, open_positions) -> None:
        """Detect positions that closed since last poll; count wins/losses."""
        live = {p["ticket"]: p["profit"] for p in open_positions}
        for ticket, last_profit in list(self.known_tickets.items()):
            if ticket not in live:  # closed
                if last_profit < 0:
                    self.losses_today += 1
                logger.trade("DETECTED_CLOSE", ticket=ticket,
                             last_profit=last_profit)
                del self.known_tickets[ticket]
        for t, pr in live.items():
            self.known_tickets[t] = pr

    def _past_hard_flat(self, now: datetime) -> bool:
        return (now.hour, now.minute) >= self.hard_flat

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        if not self.conn.connect():
            logger.error("agent cannot start: MT5 connection failed.")
            return
        mode = ("LIVE" if self.exec.live_unlocked() else "DEMO")
        logger.info(f"=== AGENT START ({mode}) ===",
                    symbol=self.cfg.get("symbol"), poll=self.poll)
        try:
            while True:
                try:
                    self._iteration()
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.error("iteration error", err=str(e))
                _time.sleep(self.poll)
        except KeyboardInterrupt:
            logger.warn("agent stopped by user (Ctrl-C)")
        finally:
            acc = self.conn.account()
            if acc:
                self._finish_day(acc["equity"])
            self.conn.shutdown()
            logger.info("=== AGENT STOPPED ===")

    def _iteration(self) -> None:
        if not self.conn.is_connected():
            logger.warn("MT5 disconnected -- skipping iteration (no trading)")
            return
        acc = self.conn.account()
        spec = self.conn.symbol_spec()
        spread = self.conn.spread_pips()
        bars = self.conn.get_bars(self.cfg.get("backtest", {})
                                  .get("history_bars", 4000))
        if acc is None or spec is None or spread is None or not bars:
            logger.warn("market data incomplete -- skipping (no trading)")
            return

        equity = acc["equity"]
        self.equity_peak = max(self.equity_peak, equity)
        now = datetime.now(timezone.utc)
        self._rollover(now, equity)

        dd = self._drawdown(equity)
        self.day_max_dd = max(self.day_max_dd, dd)
        rstate = self.recovery.evaluate(dd)

        positions = self.conn.open_positions(magic=self.magic)
        self._track_closed_positions(positions)

        # --- kill switch / shutdown --------------------------------------
        if rstate.halted:
            if positions:
                logger.warn("SHUTDOWN active -- flattening all positions")
                self.exec.flatten_all(reason="shutdown")
            logger.error("AGENT HALTED (Stage 4)", note=rstate.note)
            self._finish_day(equity)
            raise KeyboardInterrupt  # stop the loop cleanly

        # --- hard-flat exit ----------------------------------------------
        if positions and self._past_hard_flat(now):
            logger.info("hard-flat time reached -- closing positions")
            self.exec.flatten_all(reason="hard_flat")
            return

        # --- already in a trade? -----------------------------------------
        if positions:
            return

        # --- news blackout ------------------------------------------------
        blackout = False
        if self.news_enabled:
            if self.news_events is None:
                policy = (self.cfg.get("news", {})
                          .get("on_missing_calendar", "halve_risk"))
                logger.warn("news calendar missing", policy=policy)
                if policy == "skip_day":
                    return
            else:
                blackout = in_news_blackout(self.news_events, now,
                                            self.news_before, self.news_after)

        # --- permission gate ---------------------------------------------
        state = AccountState(
            equity=equity, balance=acc["balance"],
            start_of_day_equity=self.day_start_equity,
            start_of_week_equity=self.week_start_equity,
            initial_capital=self.initial_capital,
            equity_peak=self.equity_peak,
            open_trades=len(positions),
            trades_today=self.trades_today,
            losses_today=self.losses_today,
            losing_days_this_week=self.losing_days_week,
            spread_pips=spread, connected=True,
            symbol_info_ok=True, history_ok=True,
            in_news_blackout=blackout, in_session=True,
        )
        perm = self.risk.can_trade(state)
        if not perm.allowed:
            logger.decision("seek_signal", False, perm.reason)
            return
        if self.trades_today >= rstate.trades_per_day:
            logger.decision("seek_signal", False,
                            f"recovery stage limits trades/day to "
                            f"{rstate.trades_per_day}")
            return

        # --- new bar? -----------------------------------------------------
        last_bar = bars[-2] if len(bars) >= 2 else bars[-1]  # last COMPLETED
        if self.last_bar_time == last_bar.time:
            return  # already evaluated this bar
        self.last_bar_time = last_bar.time

        completed = bars[:-1] if len(bars) >= 2 else bars
        sig = self.strategy.evaluate(
            completed, confirm_closes_override=rstate.confirm_closes)
        if sig is None:
            return
        if sig.direction in self.dirs_today:
            logger.decision("enter_trade", False,
                            "direction already attempted today")
            return
        if not rstate.both_directions:
            logger.decision("enter_trade", False,
                            "recovery stage 3: single-direction only "
                            "(verify best direction from backtest)")
            return

        # --- size + execute ----------------------------------------------
        tick = self.conn.tick()
        if tick is None:
            logger.decision("enter_trade", False, "tick unreadable")
            return
        fill_ref = tick["ask"] if sig.direction > 0 else tick["bid"]
        orders = sig.build_orders(fill_ref)
        lots = self.risk.position_size(equity, orders["stop_distance"], spec,
                                       risk_fraction=rstate.risk_per_trade)
        if lots <= 0:
            logger.decision("enter_trade", False,
                            "position size resolved to 0 (risk cap vs lot min)")
            return

        logger.decision("enter_trade", True,
                        f"{sig.reason} | stage={rstate.stage} "
                        f"risk={rstate.risk_per_trade}")
        result = self.exec.open_trade(sig, lots)
        if result:
            self.trades_today += 1
            self.dirs_today.add(sig.direction)
            self.known_tickets[result["ticket"]] = 0.0


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="mt5_quant_trading_agent")
    parser.add_argument("--check", action="store_true",
                        help="run preflight only, no trading")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.check:
        ok = preflight(config)
        return 0 if ok else 1

    # full run requires preflight to pass first
    if not preflight(config):
        logger.error("preflight failed -- agent will not start.")
        return 1

    if config.get("account_mode") == "live" and config.get("live_trading_enabled"):
        logger.warn("LIVE TRADING IS UNLOCKED IN CONFIG. Real money at risk.")
        logger.warn("Ctrl-C within 10s to abort.")
        _time.sleep(10)

    TradingAgent(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
