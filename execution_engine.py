"""
execution_engine.py
-------------------
Places, modifies and closes orders -- with a hard DEMO/LIVE guard.

Live-money safety latch (ALL must be true to touch a REAL account):
  1. config account_mode == "live"
  2. config live_trading_enabled == True
  3. the terminal's account is actually a real account
If the terminal is logged into a REAL account but the config is not fully
unlocked, the engine REFUSES every order. If the account is demo, orders
flow normally (it is demo money).

No order is ever sent without a stop-loss. If the SL cannot be attached,
the order is not sent.
"""

from __future__ import annotations

from typing import Optional

from mt5_connector import MT5Connector, MT5_AVAILABLE, mt5
from strategy import Signal
from trade_logger import logger


# MT5 trade_mode constants (avoid importing if package missing)
TRADE_MODE_DEMO = 0
TRADE_MODE_CONTEST = 1
TRADE_MODE_REAL = 2


class ExecutionEngine:
    def __init__(self, config: dict, connector: MT5Connector) -> None:
        self.cfg = config
        self.conn = connector
        self.magic = int(config.get("magic_number", 770524))
        self.max_slippage_pips = float(config.get("max_slippage_pips", 0.5))
        self.account_mode = str(config.get("account_mode", "demo")).lower()
        self.live_enabled = bool(config.get("live_trading_enabled", False))

    # -- the safety latch --------------------------------------------------
    def live_unlocked(self) -> bool:
        """True only when BOTH config flags explicitly authorise live money."""
        return self.account_mode == "live" and self.live_enabled is True

    def account_is_real(self) -> Optional[bool]:
        """True/False if the terminal account is real; None if unreadable."""
        if not (MT5_AVAILABLE and self.conn.is_connected()):
            return None
        ai = mt5.account_info()
        if ai is None:
            return None
        return int(getattr(ai, "trade_mode", TRADE_MODE_DEMO)) == TRADE_MODE_REAL

    def trading_permitted(self) -> tuple[bool, str]:
        """
        Decide whether ANY order may be sent right now.
        Refuses if a real account is detected without the live latch open.
        """
        is_real = self.account_is_real()
        if is_real is None:
            return False, "account trade-mode unreadable -- refusing to trade"
        if is_real and not self.live_unlocked():
            return False, ("REAL account detected but live latch is closed "
                           "(account_mode/live_trading_enabled). Refusing.")
        if is_real and self.live_unlocked():
            return True, "LIVE trading authorised on a real account"
        return True, "demo account -- safe to trade"

    # -- order placement ---------------------------------------------------
    def open_trade(self, signal: Signal, lots: float) -> Optional[dict]:
        """
        Send a market order in the signal's direction with SL + TP attached.
        Returns a result dict, or None if the trade was refused / failed.
        """
        permitted, why = self.trading_permitted()
        if not permitted:
            logger.decision("place_order", False, why)
            return None
        if lots <= 0:
            logger.decision("place_order", False, "lot size resolved to 0")
            return None
        if not MT5_AVAILABLE:
            logger.error("cannot place order: MetaTrader5 unavailable")
            return None

        tick = self.conn.tick()
        if tick is None:
            logger.decision("place_order", False, "tick unreadable")
            return None
        fill_ref = tick["ask"] if signal.direction > 0 else tick["bid"]
        orders = signal.build_orders(fill_ref)
        sl, tp = orders["stop_loss"], orders["take_profit"]

        if sl <= 0 or (signal.direction > 0 and sl >= fill_ref) \
                or (signal.direction < 0 and sl <= fill_ref):
            logger.decision("place_order", False, "invalid stop-loss -- refused")
            return None

        spec = self.conn.symbol_spec()
        if spec is None:
            logger.decision("place_order", False, "symbol spec unreadable")
            return None
        deviation = int(self.max_slippage_pips * (spec.pip_size / mt5.symbol_info(
            self.conn.symbol).point))

        order_type = (mt5.ORDER_TYPE_BUY if signal.direction > 0
                      else mt5.ORDER_TYPE_SELL)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.conn.symbol,
            "volume": float(lots),
            "type": order_type,
            "price": fill_ref,
            "sl": round(sl, 6),
            "tp": round(tp, 6),
            "deviation": deviation,
            "magic": self.magic,
            "comment": "mt5_quant_agent",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            rc = getattr(result, "retcode", "none")
            logger.error("order_send failed", retcode=rc)
            logger.decision("place_order", False, f"order_send retcode={rc}")
            return None

        logger.trade("OPEN", direction=signal.direction, lots=lots,
                      price=result.price, sl=sl, tp=tp, ticket=result.order)
        return {"ticket": result.order, "price": result.price,
                "lots": lots, "sl": sl, "tp": tp,
                "direction": signal.direction}

    def close_position(self, position: dict, reason: str = "manual") -> bool:
        """Close an open position by ticket."""
        permitted, why = self.trading_permitted()
        if not permitted:
            logger.decision("close_order", False, why)
            return False
        if not MT5_AVAILABLE:
            return False
        tick = self.conn.tick()
        if tick is None:
            return False
        is_buy = position["type"] == 0
        price = tick["bid"] if is_buy else tick["ask"]
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.conn.symbol,
            "volume": float(position["volume"]),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": position["ticket"],
            "price": price,
            "deviation": 50,
            "magic": self.magic,
            "comment": f"close:{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        logger.trade("CLOSE" if ok else "CLOSE_FAILED",
                     ticket=position["ticket"], reason=reason)
        return ok

    def flatten_all(self, reason: str = "kill_switch") -> int:
        """Close every position with our magic number. Returns count closed."""
        closed = 0
        for pos in self.conn.open_positions(magic=self.magic):
            if self.close_position(pos, reason=reason):
                closed += 1
        logger.warn("flatten_all executed", reason=reason, closed=closed)
        return closed
