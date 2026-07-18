"""
daily_review.py
---------------
End-of-day review. Answers the 10-point checklist from research report
Part 8 and produces a continue / reduce-risk / pause verdict.

Key discipline rule encoded here: the review may ONLY recommend
"continue", "reduce risk", or "pause". It can never recommend raising
risk or changing strategy logic -- those happen in the weekly process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from report_generator import write_daily_report
from trade_logger import logger


@dataclass
class DayStats:
    date: str
    start_balance: float
    end_balance: float
    start_equity: float
    end_equity: float
    trades: int = 0
    wins: int = 0
    losses: int = 0
    net_pl: float = 0.0
    net_pl_R: float = 0.0
    max_intraday_dd: float = 0.0
    avg_spread_pips: float = 0.0
    avg_slippage_pips: float = 0.0
    rule_violations: List[str] = field(default_factory=list)
    best_trade: float = 0.0
    worst_trade: float = 0.0
    recovery_stage: int = 0
    expected_loss_streak_p95: int = 6
    current_loss_streak: int = 0
    mc_expectancy_R: float = 0.0
    live_expectancy_R: float = 0.0


def run_review(stats: DayStats, assumed_spread: float = 0.7,
               assumed_slippage: float = 0.3) -> dict:
    """Return the 10-point checklist answers + an overall verdict."""
    cl: dict = {}

    # 1. Did trades follow the rules?
    cl["1. trades followed rules"] = (
        "YES" if not stats.rule_violations
        else "NO -- " + "; ".join(stats.rule_violations))

    # 2. Spread / slippage vs assumptions
    spread_bad = stats.avg_spread_pips > assumed_spread * 1.3
    slip_bad = stats.avg_slippage_pips > assumed_slippage * 1.3
    cl["2. spread/slippage within assumptions"] = (
        "OK" if not (spread_bad or slip_bad)
        else f"EXCEEDED (spread {stats.avg_spread_pips:.2f}, "
             f"slip {stats.avg_slippage_pips:.2f})")

    # 3. Pair behaving normally
    cl["3. pair behaving normally"] = (
        "YES" if stats.max_intraday_dd < 0.04 else "ELEVATED VOLATILITY")

    # 4. Loss within expected range
    day_ret = ((stats.end_equity - stats.start_equity) / stats.start_equity
               if stats.start_equity > 0 else 0.0)
    cl["4. loss within expected range"] = (
        "YES" if day_ret > -0.03 else "NO -- daily loss limit region")

    # 5. Drawdown triggered recovery mode
    cl["5. recovery stage"] = f"Stage {stats.recovery_stage}"

    # 6/7/8. continue / reduce / pause
    reduce_risk = (stats.recovery_stage >= 1 or spread_bad or slip_bad
                   or day_ret <= -0.02)
    pause = (stats.recovery_stage >= 4
             or stats.current_loss_streak > stats.expected_loss_streak_p95
             or day_ret <= -0.03)
    cl["6. continue unchanged"] = "NO" if (reduce_risk or pause) else "YES"
    cl["7. reduce risk"] = "YES" if reduce_risk and not pause else "NO"
    cl["8. pause trading"] = "YES" if pause else "NO"

    # 9. optimisation needed?
    cl["9. optimisation needed"] = (
        "Maybe -- flag for WEEKLY review (never act mid-week)"
        if (stats.live_expectancy_R != 0.0
            and stats.live_expectancy_R < 0.5 * stats.mc_expectancy_R)
        else "No evidence yet")

    # 10. statistically meaningful?
    cl["10. statistically meaningful"] = (
        "YES" if stats.trades >= 200
        else f"NOT YET (need ~200, have running total -- today {stats.trades})")

    if pause:
        verdict = "PAUSE -- stop trading; human review before resuming."
    elif reduce_risk:
        verdict = "CONTINUE WITH REDUCED RISK -- recovery/condition rules apply."
    else:
        verdict = "CONTINUE UNCHANGED -- system within expectation."
    cl["VERDICT"] = verdict

    return cl


def end_of_day(stats: DayStats) -> str:
    """Build the day summary, run the checklist, write the report."""
    summary = {
        "date": stats.date,
        "starting balance": round(stats.start_balance, 2),
        "ending balance": round(stats.end_balance, 2),
        "starting equity": round(stats.start_equity, 2),
        "ending equity": round(stats.end_equity, 2),
        "trades": stats.trades,
        "wins": stats.wins,
        "losses": stats.losses,
        "net P/L (money)": round(stats.net_pl, 2),
        "net P/L (R)": round(stats.net_pl_R, 3),
        "max intraday drawdown": f"{stats.max_intraday_dd * 100:.2f}%",
        "avg spread (pips)": round(stats.avg_spread_pips, 2),
        "avg slippage (pips)": round(stats.avg_slippage_pips, 2),
        "best trade (money)": round(stats.best_trade, 2),
        "worst trade (money)": round(stats.worst_trade, 2),
        "recovery stage": stats.recovery_stage,
        "rule violations": "; ".join(stats.rule_violations) or "none",
    }
    checklist = run_review(stats)
    path = write_daily_report(summary, checklist, stats.date)
    logger.info("daily review written", path=path, verdict=checklist["VERDICT"])
    return path
