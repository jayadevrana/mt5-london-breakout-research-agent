"""
trade_logger.py
---------------
Structured logging of EVERY decision the agent makes. The point of this
module is auditability: after any trading day you can reconstruct exactly
why the agent did or did not act.

Two outputs:
  * logs/agent_YYYY-MM-DD.log   -- human-readable line log
  * logs/decisions_YYYY-MM-DD.jsonl -- one JSON object per decision

No third-party dependencies.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _ensure_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class TradeLogger:
    """Append-only logger. Safe to construct many times in one process."""

    LEVELS = ("DEBUG", "INFO", "WARN", "ERROR", "DECISION", "TRADE")

    def __init__(self, echo: bool = True) -> None:
        _ensure_dir()
        self.echo = echo

    # -- low level ---------------------------------------------------------
    def _line_path(self) -> str:
        return os.path.join(LOG_DIR, f"agent_{_today()}.log")

    def _json_path(self) -> str:
        return os.path.join(LOG_DIR, f"decisions_{_today()}.jsonl")

    def log(self, level: str, message: str, **fields) -> None:
        level = level.upper()
        if level not in self.LEVELS:
            level = "INFO"
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"{ts} [{level:8}] {message}" + (f" | {extra}" if extra else "")
        try:
            with open(self._line_path(), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
        if self.echo:
            print(line)

    # -- convenience -------------------------------------------------------
    def info(self, msg: str, **f) -> None:
        self.log("INFO", msg, **f)

    def warn(self, msg: str, **f) -> None:
        self.log("WARN", msg, **f)

    def error(self, msg: str, **f) -> None:
        self.log("ERROR", msg, **f)

    def debug(self, msg: str, **f) -> None:
        self.log("DEBUG", msg, **f)

    # -- structured decision record ---------------------------------------
    def decision(self, what: str, allowed: bool, reason: str, **context) -> None:
        """Record a yes/no decision (e.g. 'enter_long', 'place_order')."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "type": "decision",
            "what": what,
            "allowed": bool(allowed),
            "reason": reason,
            "context": context,
        }
        self._write_json(record)
        self.log("DECISION", f"{what}: {'ALLOW' if allowed else 'BLOCK'} -- {reason}")

    def trade(self, action: str, **details) -> None:
        """Record an executed trade event (open/close/modify)."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "type": "trade",
            "action": action,
            "details": details,
        }
        self._write_json(record)
        self.log("TRADE", f"{action}", **details)

    def _write_json(self, record: dict) -> None:
        try:
            with open(self._json_path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass


# module-level singleton for convenience
logger = TradeLogger()
