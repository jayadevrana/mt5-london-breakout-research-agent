"""
report_generator.py
-------------------
Produces the daily report and the backtest report as plain text + HTML.
Reports land in reports/.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _ensure() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)


def _kv_text(d: dict) -> str:
    return "\n".join(f"  {k:30} {v}" for k, v in d.items())


def _kv_html(d: dict) -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>"
        for k, v in d.items())
    return f"<table border='1' cellpadding='6' cellspacing='0'>{rows}</table>"


def write_daily_report(summary: dict, checklist: dict,
                        date_str: str | None = None) -> str:
    """summary: the day's numbers. checklist: the 10-point review answers."""
    _ensure()
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base = os.path.join(REPORT_DIR, f"daily_{date_str}")

    text = (
        f"END-OF-DAY REPORT  --  {date_str}\n"
        f"{'=' * 52}\n\n"
        f"DAY SUMMARY\n{_kv_text(summary)}\n\n"
        f"REVIEW CHECKLIST\n{_kv_text(checklist)}\n\n"
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"Reminder: do NOT modify the EA on emotion. Daily review may only\n"
        f"reduce risk or pause. Logic changes happen in the weekly process.\n"
    )
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text)

    html = (
        f"<html><head><meta charset='utf-8'><title>EOD {date_str}</title>"
        f"<style>body{{font-family:system-ui,Arial;margin:32px}}"
        f"td{{font-size:14px}}</style></head><body>"
        f"<h2>End-of-day report &mdash; {date_str}</h2>"
        f"<h3>Day summary</h3>{_kv_html(summary)}"
        f"<h3>Review checklist</h3>{_kv_html(checklist)}"
        f"<p style='color:#666'>Do not modify the EA on emotion. Daily review "
        f"may only reduce risk or pause.</p></body></html>"
    )
    with open(base + ".html", "w", encoding="utf-8") as fh:
        fh.write(html)
    return base + ".txt"


def write_backtest_report(metrics: dict, stressed: dict | None = None,
                           extra: dict | None = None) -> str:
    """Write a backtest metrics report."""
    _ensure()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORT_DIR, f"backtest_{stamp}.txt")
    parts = [f"BACKTEST REPORT  --  {stamp}", "=" * 52, "",
             "METRICS", _kv_text(metrics)]
    if stressed:
        parts += ["", "COST-STRESSED (+0.5 spread / +0.3 slippage)",
                  _kv_text(stressed)]
    if extra:
        parts += ["", "ADDITIONAL", _kv_text(extra)]
    parts += ["", "Acceptance gate is in optimizer.py. Passing the backtest "
              "alone is NOT sufficient to trade."]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts) + "\n")
    return path
