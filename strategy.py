"""
strategy.py
-----------
The selected edge: compressed Asian-range, London-session breakout
continuation (research report H1 + ATR-compression gate + close
confirmation). EURUSD M15 by default.

This module is PURE LOGIC. It works on a plain list of Bar objects and
imports nothing from MetaTrader5, so it can be unit-tested and backtested
with no terminal present.

It makes NO directional forecast. Its only claim: a compressed overnight
range, broken on a CLOSING basis at the London liquidity influx, continues
far enough often enough to clear costs. That claim is what the backtester
and optimizer test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import List, Optional


# --------------------------------------------------------------------------
# data structures
# --------------------------------------------------------------------------
@dataclass
class Bar:
    time: datetime          # bar OPEN time, broker/server timezone
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class Signal:
    direction: int          # +1 long, -1 short
    ref_price: float        # close of the confirming bar (reference only)
    atr: float
    asia_high: float
    asia_low: float
    asia_range: float
    stop_atr_mult: float
    reward_mult: float
    quality: str            # "A+" or "standard"
    reason: str

    def build_orders(self, fill_price: float) -> dict:
        """
        Given the ACTUAL fill price (decided by engine/backtester), return
        the stop-loss and take-profit. SL distance = min(ATR-based,
        opposite-Asian-range-side) so the stop is always structurally
        justified and never wider than the range logic implies.
        """
        atr_dist = self.stop_atr_mult * self.atr
        if self.direction > 0:
            range_dist = max(1e-9, fill_price - self.asia_low)
            sl_dist = min(atr_dist, range_dist)
            sl = fill_price - sl_dist
            tp = fill_price + self.reward_mult * sl_dist
        else:
            range_dist = max(1e-9, self.asia_high - fill_price)
            sl_dist = min(atr_dist, range_dist)
            sl = fill_price + sl_dist
            tp = fill_price - self.reward_mult * sl_dist
        return {"stop_loss": sl, "take_profit": tp, "stop_distance": sl_dist}


# --------------------------------------------------------------------------
# indicator helpers
# --------------------------------------------------------------------------
def true_range(prev_close: float, high: float, low: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr(bars: List[Bar], period: int) -> float:
    """Simple-average ATR over the last `period` completed bars."""
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(len(bars) - period, len(bars)):
        trs.append(true_range(bars[i - 1].close, bars[i].high, bars[i].low))
    return sum(trs) / period if trs else 0.0


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _day_ranges(bars: List[Bar], exclude_day: date, lookback_days: int) -> List[float]:
    """High-Low range of each completed prior day, most recent `lookback_days`."""
    by_day: dict = {}
    for b in bars:
        d = b.time.date()
        if d == exclude_day:
            continue
        hi, lo = by_day.get(d, (b.high, b.low))
        by_day[d] = (max(hi, b.high), min(lo, b.low))
    ranges = [hi - lo for _, (hi, lo) in sorted(by_day.items())]
    return ranges[-lookback_days:] if lookback_days else ranges


# --------------------------------------------------------------------------
# strategy
# --------------------------------------------------------------------------
class Strategy:
    def __init__(self, config: dict) -> None:
        s = config.get("strategy", {}) or {}
        self.atr_period = int(s.get("atr_period", 14))
        self.compression_k = float(s.get("compression_k", 0.55))
        self.breakout_buffer_atr = float(s.get("breakout_buffer_atr", 0.10))
        self.stop_atr_mult = float(s.get("stop_atr_mult", 1.10))
        self.reward_mult = float(s.get("reward_mult", 1.60))
        self.confirm_closes = int(s.get("confirm_closes", 1))
        self.asia_start = _parse_hhmm(config.get("asia_start", "00:00"))
        self.asia_end = _parse_hhmm(config.get("asia_end", "07:00"))
        self.session_start = _parse_hhmm(config.get("session_start", "07:00"))
        self.session_end = _parse_hhmm(config.get("session_end", "11:00"))
        self.daily_lookback = 20

    # -- the one public method -------------------------------------------
    def evaluate(self, bars: List[Bar],
                 confirm_closes_override: Optional[int] = None) -> Optional[Signal]:
        """
        Decide whether the MOST RECENT completed bar (bars[-1]) completes a
        fresh, confirmed Asian-range breakout inside the London window.
        Returns a Signal or None. The engine handles per-day deduplication.
        """
        n = len(bars)
        if n < self.atr_period + 5:
            return None

        confirm = confirm_closes_override or self.confirm_closes
        last = bars[-1]
        today = last.time.date()
        t = last.time.time()

        # must be inside the London trading window
        if not (self.session_start <= t < self.session_end):
            return None

        # build the Asian range for today; require it to be complete
        asia = [b for b in bars
                if b.time.date() == today
                and self.asia_start <= b.time.time() < self.asia_end]
        if len(asia) < 3:
            return None
        asia_high = max(b.high for b in asia)
        asia_low = min(b.low for b in asia)
        asia_range = asia_high - asia_low
        if asia_range <= 0:
            return None

        # compression gate -- the Asian range must be genuinely tight
        day_ranges = _day_ranges(bars, today, self.daily_lookback)
        if len(day_ranges) < 5:
            return None
        daily_atr_avg = sum(day_ranges) / len(day_ranges)
        if daily_atr_avg <= 0:
            return None
        compressed = asia_range <= self.compression_k * daily_atr_avg
        if not compressed:
            return None  # not a tradeable day

        a = atr(bars, self.atr_period)
        if a <= 0:
            return None
        buffer = self.breakout_buffer_atr * a
        long_level = asia_high + buffer
        short_level = asia_low - buffer

        # confirmation: the last `confirm` bars must all close beyond the
        # level, AND the bar immediately before that run must NOT be beyond
        # it (so this is a FRESH breakout, not a re-signal).
        if n < confirm + 1:
            return None
        run = bars[-confirm:]
        pre = bars[-confirm - 1]

        long_ok = all(b.close > long_level for b in run) and pre.close <= long_level
        short_ok = all(b.close < short_level for b in run) and pre.close >= short_level

        # also: the breakout run must occur within the London window
        if not all(self.session_start <= b.time.time() < self.session_end
                   for b in run):
            return None

        quality = "A+" if asia_range <= 0.45 * daily_atr_avg else "standard"

        if long_ok and not short_ok:
            return Signal(direction=1, ref_price=last.close, atr=a,
                          asia_high=asia_high, asia_low=asia_low,
                          asia_range=asia_range,
                          stop_atr_mult=self.stop_atr_mult,
                          reward_mult=self.reward_mult, quality=quality,
                          reason=f"long breakout: close {last.close:.5f} > "
                                 f"level {long_level:.5f}, range compressed "
                                 f"({asia_range:.5f} <= {self.compression_k}*"
                                 f"{daily_atr_avg:.5f})")
        if short_ok and not long_ok:
            return Signal(direction=-1, ref_price=last.close, atr=a,
                          asia_high=asia_high, asia_low=asia_low,
                          asia_range=asia_range,
                          stop_atr_mult=self.stop_atr_mult,
                          reward_mult=self.reward_mult, quality=quality,
                          reason=f"short breakout: close {last.close:.5f} < "
                                 f"level {short_level:.5f}, range compressed "
                                 f"({asia_range:.5f} <= {self.compression_k}*"
                                 f"{daily_atr_avg:.5f})")
        return None
