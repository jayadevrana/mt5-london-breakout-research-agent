"""
backtester.py
-------------
Event-driven backtest of the Asian-range / London-breakout strategy.

Honesty features baked in:
  * round-trip cost (spread + commission-equivalent) deducted every trade
  * slippage haircut deducted every trade
  * intrabar SL-before-TP assumption (conservative -- worst case)
  * entry filled at the NEXT bar open, never the signal bar's close
  * hard-flat exit at session end
  * risk/recovery rules applied exactly as in live logic

No MetaTrader5 import -> runs anywhere. Feed it bars from MT5 or a CSV.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import List, Optional

from strategy import Bar, Strategy
from risk_manager import RiskManager, SymbolSpec
from recovery_manager import RecoveryManager


# --------------------------------------------------------------------------
@dataclass
class BTTrade:
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry: float
    exit: float
    stop_distance: float
    lots: float
    gross_pips: float
    net_pips: float
    pnl_money: float
    r_multiple: float
    outcome: str          # "win" | "loss" | "flat"
    quality: str


@dataclass
class BacktestResult:
    trades: List[BTTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    starting_balance: float = 1000.0
    metrics: dict = field(default_factory=dict)


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


# --------------------------------------------------------------------------
class Backtester:
    def __init__(self, config: dict, symbol: Optional[SymbolSpec] = None) -> None:
        self.cfg = config
        self.strategy = Strategy(config)
        self.risk = RiskManager(config)
        self.recovery = RecoveryManager(config)
        bt = config.get("backtest", {}) or {}
        self.spread_pips = float(bt.get("spread_pips", 0.7))
        self.slippage_pips = float(bt.get("slippage_pips", 0.3))
        self.start_balance = float(bt.get("starting_balance", 1000.0))
        self.hard_flat = _parse_hhmm(config.get("hard_flat_time", "20:00"))
        self.symbol = symbol or SymbolSpec()  # EURUSD-like defaults

    # -- run ---------------------------------------------------------------
    def run(self, bars: List[Bar],
            spread_override: Optional[float] = None,
            slippage_override: Optional[float] = None) -> BacktestResult:
        spread = self.spread_pips if spread_override is None else spread_override
        slip = self.slippage_pips if slippage_override is None else slippage_override
        cost_price = (spread + slip) * self.symbol.pip_size

        equity = balance = self.start_balance
        initial = self.start_balance
        peak = equity
        res = BacktestResult(starting_balance=self.start_balance)
        res.equity_curve.append(equity)

        open_trade: Optional[dict] = None
        cur_day = None
        cur_week = None
        trades_today = 0
        losses_today = 0
        dirs_today: set = set()
        losing_days_week = 0
        day_start_equity = equity

        warmup = max(self.strategy.atr_period + 5, 30 * 20)  # ~30 days of M15

        for i in range(warmup, len(bars) - 1):
            bar = bars[i]
            nxt = bars[i + 1]
            d = bar.time.date()
            wk = bar.time.isocalendar()[1]

            # --- day / week rollover ------------------------------------
            if d != cur_day:
                if cur_day is not None and losses_today > 0:
                    losing_days_week += 1
                cur_day = d
                trades_today = 0
                losses_today = 0
                dirs_today = set()
                day_start_equity = equity
            if wk != cur_week:
                cur_week = wk
                losing_days_week = 0

            # --- manage an open trade -----------------------------------
            if open_trade is not None:
                exit_price, exit_reason = self._check_exit(open_trade, bar)
                if exit_price is not None:
                    t = self._close_trade(open_trade, bar.time, exit_price,
                                           cost_price, balance)
                    res.trades.append(t)
                    balance += t.pnl_money
                    equity = balance
                    peak = max(peak, equity)
                    if t.outcome == "loss":
                        losses_today += 1
                    open_trade = None

            # --- equity curve point -------------------------------------
            res.equity_curve.append(equity)

            # --- recovery / drawdown ------------------------------------
            dd = self._drawdown(equity, initial, peak)
            rstate = self.recovery.evaluate(dd)
            if rstate.halted:
                continue  # Stage 4 -- no trading

            # --- can we look for a trade? -------------------------------
            if open_trade is not None:
                continue
            if trades_today >= min(self.risk.max_trades_per_day,
                                   rstate.trades_per_day):
                continue
            if losses_today >= self.risk.max_losses_per_day:
                continue
            day_loss = ((day_start_equity - equity) / day_start_equity
                        if day_start_equity > 0 else 0.0)
            if day_loss >= self.risk.max_daily_loss:
                continue
            if losing_days_week >= self.risk.max_losing_days:
                continue

            # --- signal -------------------------------------------------
            # Pass a BOUNDED window (~25 days) so each evaluation is O(window)
            # instead of O(i): keeps a full multi-year backtest fast.
            lo = max(0, i - 2400)
            sig = self.strategy.evaluate(bars[lo:i + 1],
                                         confirm_closes_override=rstate.confirm_closes)
            if sig is None:
                continue
            if sig.direction in dirs_today:
                continue  # one attempt per direction per day
            if not rstate.both_directions and sig.direction != self._best_dir():
                continue  # Stage 3: only the historically best direction

            # --- size + enter at next bar open --------------------------
            fill = nxt.open + (cost_price / 2.0) * (1 if sig.direction > 0 else -1)
            orders = sig.build_orders(fill)
            lots = self.risk.position_size(equity, orders["stop_distance"],
                                           self.symbol,
                                           risk_fraction=rstate.risk_per_trade)
            if lots <= 0:
                continue
            open_trade = {
                "signal": sig, "direction": sig.direction, "entry": fill,
                "sl": orders["stop_loss"], "tp": orders["take_profit"],
                "stop_distance": orders["stop_distance"], "lots": lots,
                "entry_time": nxt.time, "quality": sig.quality,
            }
            trades_today += 1
            dirs_today.add(sig.direction)

        # close any still-open trade at the last bar
        if open_trade is not None:
            last = bars[-1]
            t = self._close_trade(open_trade, last.time, last.close,
                                   cost_price, balance)
            res.trades.append(t)
            balance += t.pnl_money
            res.equity_curve.append(balance)

        res.metrics = compute_metrics(res.trades, res.equity_curve,
                                      self.start_balance)
        return res

    # -- helpers -----------------------------------------------------------
    def _best_dir(self) -> int:
        """Historically favoured direction for Stage-3 recovery. Default long.
        Override after your own backtest tells you which side is stronger."""
        return 1

    def _check_exit(self, tr: dict, bar: Bar):
        """Return (exit_price, reason) or (None, None). SL assumed hit first."""
        if tr["direction"] > 0:
            if bar.low <= tr["sl"]:
                return tr["sl"], "stop"
            if bar.high >= tr["tp"]:
                return tr["tp"], "target"
        else:
            if bar.high >= tr["sl"]:
                return tr["sl"], "stop"
            if bar.low <= tr["tp"]:
                return tr["tp"], "target"
        if bar.time.time() >= self.hard_flat:
            return bar.close, "hard_flat"
        return None, None

    def _close_trade(self, tr: dict, exit_time: datetime, exit_price: float,
                     cost_price: float, balance: float) -> BTTrade:
        direction = tr["direction"]
        gross = (exit_price - tr["entry"]) * direction
        net = gross - cost_price  # remaining half-cost on exit
        ps = self.symbol.pip_size
        gross_pips = gross / ps
        net_pips = net / ps
        pnl_money = (net / ps) * self.symbol.pip_value_per_lot * tr["lots"]
        r = net / tr["stop_distance"] if tr["stop_distance"] > 0 else 0.0
        outcome = "win" if pnl_money > 0 else ("loss" if pnl_money < 0 else "flat")
        return BTTrade(
            entry_time=tr["entry_time"], exit_time=exit_time,
            direction=direction, entry=tr["entry"], exit=exit_price,
            stop_distance=tr["stop_distance"], lots=tr["lots"],
            gross_pips=gross_pips, net_pips=net_pips, pnl_money=pnl_money,
            r_multiple=r, outcome=outcome, quality=tr["quality"],
        )

    @staticmethod
    def _drawdown(equity: float, initial: float, peak: float) -> float:
        dd_i = (initial - equity) / initial if initial > 0 else 0.0
        dd_p = (peak - equity) / peak if peak > 0 else 0.0
        return max(0.0, dd_i, dd_p)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def compute_metrics(trades: List[BTTrade], equity_curve: List[float],
                     starting_balance: float) -> dict:
    """Every performance number the research report (Part 3C) requires."""
    n = len(trades)
    if n == 0:
        return {"trades": 0, "note": "no trades -- nothing to evaluate"}

    wins = [t for t in trades if t.pnl_money > 0]
    losses = [t for t in trades if t.pnl_money < 0]
    gross_win = sum(t.pnl_money for t in wins)
    gross_loss = -sum(t.pnl_money for t in losses)
    net = sum(t.pnl_money for t in trades)

    win_rate = len(wins) / n
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    loss_rate = len(losses) / n
    expectancy_money = (win_rate * avg_win) - (loss_rate * avg_loss)
    avg_r = sum(t.r_multiple for t in trades) / n
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else math.inf

    # drawdown from the equity curve
    peak = equity_curve[0] if equity_curve else starting_balance
    max_dd = 0.0
    dd_samples = []
    for e in equity_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0.0
        dd_samples.append(dd)
        max_dd = max(max_dd, dd)
    avg_dd = sum(dd_samples) / len(dd_samples) if dd_samples else 0.0

    # streaks
    win_streak = loss_streak = cur_w = cur_l = 0
    for t in trades:
        if t.pnl_money > 0:
            cur_w += 1
            cur_l = 0
        elif t.pnl_money < 0:
            cur_l += 1
            cur_w = 0
        win_streak = max(win_streak, cur_w)
        loss_streak = max(loss_streak, cur_l)

    # per-trade R series stats -> Sharpe/Sortino (per-trade basis)
    rs = [t.r_multiple for t in trades]
    mean_r = sum(rs) / n
    var = sum((x - mean_r) ** 2 for x in rs) / n
    std = math.sqrt(var)
    downside = [x for x in rs if x < 0]
    dvar = sum(x * x for x in downside) / n if downside else 0.0
    dstd = math.sqrt(dvar)
    sharpe = (mean_r / std * math.sqrt(n)) if std > 0 else 0.0
    sortino = (mean_r / dstd * math.sqrt(n)) if dstd > 0 else 0.0

    total_return = net / starting_balance
    mar = (total_return / max_dd) if max_dd > 0 else math.inf
    recovery_factor = (net / (max_dd * starting_balance)) if max_dd > 0 else math.inf

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 4),
        "avg_win_money": round(avg_win, 2),
        "avg_loss_money": round(avg_loss, 2),
        "expectancy_money": round(expectancy_money, 4),
        "expectancy_R": round(avg_r, 4),
        "avg_R_multiple": round(avg_r, 4),
        "profit_factor": round(profit_factor, 3) if profit_factor != math.inf else "inf",
        "net_profit": round(net, 2),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "avg_drawdown": round(avg_dd, 4),
        "sharpe_per_trade_basis": round(sharpe, 3),
        "sortino_per_trade_basis": round(sortino, 3),
        "mar_ratio": round(mar, 3) if mar != math.inf else "inf",
        "recovery_factor": round(recovery_factor, 3) if recovery_factor != math.inf else "inf",
        "longest_win_streak": win_streak,
        "longest_loss_streak": loss_streak,
        "ending_balance": round(starting_balance + net, 2),
    }


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_bars_csv(path: str) -> List[Bar]:
    """
    Load bars from a CSV with a header. Accepted columns (case-insensitive):
      time/date, open, high, low, close, [volume]
    `time` may be 'YYYY-MM-DD HH:MM[:SS]' or 'YYYY.MM.DD HH:MM' (MT5 export).
    """
    bars: List[Bar] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        tcol = cols.get("time") or cols.get("date") or cols.get("datetime")
        for row in reader:
            raw = row[tcol].strip().replace(".", "-", 2)
            dt = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                        "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                continue
            bars.append(Bar(
                time=dt,
                open=float(row[cols["open"]]),
                high=float(row[cols["high"]]),
                low=float(row[cols["low"]]),
                close=float(row[cols["close"]]),
                volume=float(row[cols.get("volume", cols.get("open"))] or 0)
                if "volume" in cols else 0.0,
            ))
    bars.sort(key=lambda b: b.time)
    return bars


def load_config(path: str = "config.yaml") -> dict:
    import yaml  # local import so tests need not install yaml unless used
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_bars_from_mt5(config: dict) -> Optional[List[Bar]]:
    try:
        from mt5_connector import MT5Connector
        conn = MT5Connector(config)
        if not conn.connect():
            return None
        count = int(config.get("backtest", {}).get("history_bars", 60000))
        bars = conn.get_bars(count)
        conn.shutdown()
        return bars
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    cfg = load_config()
    bars = _get_bars_from_mt5(cfg)
    src = "MT5 terminal"
    if not bars:
        # fall back to a CSV the user exports from MT5 (History Center)
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", f"{cfg.get('symbol','EURUSD')}_"
                                f"{cfg.get('timeframe','M15')}.csv")
        if os.path.exists(csv_path):
            bars = load_bars_csv(csv_path)
            src = csv_path
        else:
            print("No data: MT5 not reachable and no CSV at", csv_path)
            print("Export EURUSD M15 history from MT5 and save it there.")
            return

    print(f"Loaded {len(bars)} bars from {src}")
    bt = Backtester(cfg)
    result = bt.run(bars)
    print("\n=== BACKTEST METRICS ===")
    for k, v in result.metrics.items():
        print(f"  {k:28} {v}")

    # cost stress test
    print("\n=== COST STRESS (+0.5 spread, +0.3 slippage) ===")
    stressed = bt.run(bars,
                      spread_override=bt.spread_pips + 0.5,
                      slippage_override=bt.slippage_pips + 0.3)
    for k in ("trades", "profit_factor", "expectancy_R", "net_profit",
              "max_drawdown"):
        print(f"  {k:28} {stressed.metrics.get(k)}")


if __name__ == "__main__":
    main()
