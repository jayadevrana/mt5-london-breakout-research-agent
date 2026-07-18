"""Unit tests: ATR helper + the Asian-range / London-breakout signal logic."""

from datetime import datetime, timedelta

from strategy import Bar, Strategy, atr, true_range


def cfg():
    return {
        "asia_start": "00:00", "asia_end": "07:00",
        "session_start": "07:00", "session_end": "11:00",
        "hard_flat_time": "20:00",
        "strategy": {
            "atr_period": 14, "compression_k": 0.55,
            "breakout_buffer_atr": 0.10, "stop_atr_mult": 1.10,
            "reward_mult": 1.60, "confirm_closes": 1,
        },
    }


def _bar(dt, o, h, l, c):
    return Bar(time=dt, open=o, high=h, low=l, close=c, volume=1.0)


def build_bars(asia_high=1.1030, asia_low=1.1000, breakout=True):
    """
    22 prior days (each spanning a ~100-pip daily range) + a 'today' with a
    compressed 30-pip Asian range and an optional London breakout.
    """
    bars = []
    today = datetime(2026, 4, 1)

    # prior days: 6 bars each, daily range 1.0950..1.1050 (100 pips)
    for d in range(22, 0, -1):
        day = today - timedelta(days=d)
        for j, (o, h, l, c) in enumerate([
            (1.1000, 1.1010, 1.0990, 1.1005),
            (1.1005, 1.1050, 1.1000, 1.1040),
            (1.1040, 1.1045, 1.0950, 1.0960),
            (1.0960, 1.1000, 1.0955, 1.0995),
            (1.0995, 1.1020, 1.0990, 1.1010),
            (1.1010, 1.1015, 1.1000, 1.1005),
        ]):
            bars.append(_bar(day + timedelta(hours=2 * j), o, h, l, c))

    # today's Asian session 00:00..06:45 -- compressed 30-pip range
    t = today
    while t.hour < 7:
        # oscillate between asia_low and asia_high
        mid = (asia_high + asia_low) / 2
        hi = asia_high if t.hour % 2 == 0 else mid + 0.0005
        lo = asia_low if t.hour % 2 == 1 else mid - 0.0005
        bars.append(_bar(t, mid, hi, lo, mid))
        t += timedelta(minutes=15)

    # today's London session 07:00..07:45
    level = asia_high + 0.10 * 0.0005  # buffer ~ 0.10 * ATR (ATR ~5 pips)
    london = [
        (1.1025, 1.1028, 1.1023, 1.1026),   # 07:00 inside range
        (1.1026, 1.1029, 1.1024, 1.1027),   # 07:15 inside range
        (1.1027, 1.1030, 1.1025, 1.1028),   # 07:30 still <= asia_high
    ]
    if breakout:
        london.append((1.1028, 1.1040, 1.1027, 1.1038))  # 07:45 closes ABOVE
    else:
        london.append((1.1028, 1.1031, 1.1026, 1.1029))  # 07:45 no breakout
    for j, (o, h, l, c) in enumerate(london):
        bars.append(_bar(today + timedelta(hours=7, minutes=15 * j), o, h, l, c))
    return bars, level


# -- ATR helpers -----------------------------------------------------------
def test_true_range():
    assert abs(true_range(1.1000, 1.1010, 1.0995) - 0.0015) < 1e-9
    # gap up: prev_close far below current low
    assert abs(true_range(1.0900, 1.1010, 1.1000) - 0.0110) < 1e-9


def test_atr_positive():
    bars, _ = build_bars()
    a = atr(bars, 14)
    assert a > 0


def test_atr_insufficient_bars():
    bars = [_bar(datetime(2026, 4, 1), 1.1, 1.1, 1.1, 1.1)]
    assert atr(bars, 14) == 0.0


# -- signal logic ----------------------------------------------------------
def test_long_breakout_signals():
    bars, _ = build_bars(breakout=True)
    sig = Strategy(cfg()).evaluate(bars)
    assert sig is not None
    assert sig.direction == 1
    assert sig.asia_high == 1.1030
    assert sig.asia_low == 1.1000


def test_no_signal_without_breakout():
    bars, _ = build_bars(breakout=False)
    assert Strategy(cfg()).evaluate(bars) is None


def test_no_signal_when_range_not_compressed():
    # asian range 200 pips -> exceeds compression_k * daily_atr_avg
    bars, _ = build_bars(asia_high=1.1100, asia_low=1.0900, breakout=True)
    assert Strategy(cfg()).evaluate(bars) is None


def test_build_orders_long_risk_reward():
    bars, _ = build_bars(breakout=True)
    sig = Strategy(cfg()).evaluate(bars)
    fill = 1.1038
    orders = sig.build_orders(fill)
    # long: SL below entry, TP above entry
    assert orders["stop_loss"] < fill < orders["take_profit"]
    # reward distance ~= reward_mult * stop distance
    risk = fill - orders["stop_loss"]
    reward = orders["take_profit"] - fill
    assert abs(reward - 1.60 * risk) < 1e-6


def test_outside_session_no_signal():
    bars, _ = build_bars(breakout=True)
    # shift the confirming bar to 13:00 (outside the 07:00-11:00 window)
    bars[-1] = Bar(time=datetime(2026, 4, 1, 13, 0),
                   open=1.1028, high=1.1040, low=1.1027, close=1.1038)
    assert Strategy(cfg()).evaluate(bars) is None
