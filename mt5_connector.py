"""
mt5_connector.py
----------------
Safe wrappers around the MetaTrader5 Python package.

Every function FAILS CLOSED: if anything cannot be read, it returns None
or raises a clear error, and the caller (risk manager / engine) treats
that as "do not trade". The MetaTrader5 import is guarded so the rest of
the project (logic, tests, backtester) loads on any OS without the
terminal installed.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from strategy import Bar
from risk_manager import SymbolSpec
from trade_logger import logger

try:                                  # guarded import
    import MetaTrader5 as mt5         # type: ignore
    MT5_AVAILABLE = True
except Exception:                     # noqa: BLE001
    mt5 = None                        # type: ignore
    MT5_AVAILABLE = False


_TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


class MT5Connector:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.symbol = config.get("symbol", "EURUSD")
        self.timeframe_name = config.get("timeframe", "M15")
        self.connected = False

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> bool:
        """Initialise the terminal connection. Returns False on any failure."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package not available on this system.")
            return False
        path = self.cfg.get("mt5_terminal_path") or None
        login = int(self.cfg.get("mt5_login", 0) or 0)
        kwargs = {}
        if path:
            kwargs["path"] = path
        ok = mt5.initialize(**kwargs)
        if not ok:
            logger.error("mt5.initialize failed", err=mt5.last_error())
            return False
        # optional explicit login; 0 = use the terminal's current account
        if login:
            pw = self.cfg.get("mt5_password", "")
            srv = self.cfg.get("mt5_server", "")
            if not mt5.login(login, password=pw, server=srv):
                logger.error("mt5.login failed", err=mt5.last_error())
                mt5.shutdown()
                return False
        # make sure the symbol is selected/visible
        if not mt5.symbol_select(self.symbol, True):
            logger.error("symbol_select failed", symbol=self.symbol)
            mt5.shutdown()
            return False
        self.connected = True
        logger.info("MT5 connected", symbol=self.symbol, tf=self.timeframe_name)
        return True

    def shutdown(self) -> None:
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
        self.connected = False

    def is_connected(self) -> bool:
        if not (MT5_AVAILABLE and self.connected):
            return False
        try:
            return mt5.terminal_info() is not None
        except Exception:  # noqa: BLE001
            return False

    # -- account -----------------------------------------------------------
    def account(self) -> Optional[dict]:
        """Return dict with balance/equity/margin or None if unreadable."""
        if not self.is_connected():
            return None
        ai = mt5.account_info()
        if ai is None:
            logger.error("account_info unreadable")
            return None
        return {
            "balance": float(ai.balance),
            "equity": float(ai.equity),
            "margin": float(ai.margin),
            "margin_free": float(ai.margin_free),
            "currency": ai.currency,
            "trade_allowed": bool(getattr(ai, "trade_allowed", True)),
        }

    # -- symbol ------------------------------------------------------------
    def symbol_spec(self) -> Optional[SymbolSpec]:
        """Return SymbolSpec (pip size, pip value, volume limits) or None."""
        if not self.is_connected():
            return None
        si = mt5.symbol_info(self.symbol)
        if si is None:
            logger.error("symbol_info unreadable", symbol=self.symbol)
            return None
        # pip size: a 'pip' is 10 points for 5/3-digit FX symbols
        point = si.point
        digits = si.digits
        pip_size = point * (10 if digits in (3, 5) else 1)
        # pip value per 1.0 lot in account currency
        tick_value = si.trade_tick_value
        tick_size = si.trade_tick_size or point
        pip_value_per_lot = tick_value * (pip_size / tick_size) if tick_size else 10.0
        return SymbolSpec(
            pip_size=pip_size,
            pip_value_per_lot=pip_value_per_lot,
            volume_min=float(si.volume_min),
            volume_max=float(si.volume_max),
            volume_step=float(si.volume_step),
        )

    def spread_pips(self) -> Optional[float]:
        """Current spread in pips, or None if unreadable."""
        if not self.is_connected():
            return None
        si = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        if si is None or tick is None:
            return None
        point = si.point
        digits = si.digits
        pip_size = point * (10 if digits in (3, 5) else 1)
        spread_price = tick.ask - tick.bid
        return spread_price / pip_size if pip_size else None

    def tick(self) -> Optional[dict]:
        if not self.is_connected():
            return None
        tk = mt5.symbol_info_tick(self.symbol)
        if tk is None:
            return None
        return {"bid": float(tk.bid), "ask": float(tk.ask), "time": tk.time}

    # -- history -----------------------------------------------------------
    def get_bars(self, count: int, timeframe: Optional[str] = None
                 ) -> Optional[List[Bar]]:
        """Pull the most recent `count` completed bars as Bar objects."""
        if not self.is_connected():
            return None
        tfname = timeframe or self.timeframe_name
        tf_attr = _TIMEFRAME_MAP.get(tfname.upper())
        if tf_attr is None:
            logger.error("unknown timeframe", tf=tfname)
            return None
        tf = getattr(mt5, tf_attr)
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.error("history missing / empty", symbol=self.symbol)
            return None
        bars: List[Bar] = []
        for r in rates:
            bars.append(Bar(
                time=datetime.utcfromtimestamp(int(r["time"])),
                open=float(r["open"]), high=float(r["high"]),
                low=float(r["low"]), close=float(r["close"]),
                volume=float(r["tick_volume"]),
            ))
        return bars

    # -- open positions ----------------------------------------------------
    def open_positions(self, magic: Optional[int] = None) -> List[dict]:
        """Positions for this symbol, optionally filtered to our magic id."""
        if not self.is_connected():
            return []
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        out = []
        for p in positions:
            if magic is not None and p.magic != magic:
                continue
            out.append({
                "ticket": p.ticket, "type": p.type, "volume": float(p.volume),
                "price_open": float(p.price_open), "sl": float(p.sl),
                "tp": float(p.tp), "profit": float(p.profit), "magic": p.magic,
            })
        return out
