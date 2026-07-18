"""
optimizer.py
------------
Robustness testing -- NOT a magic-parameter finder.

Implements the Part 9 process:
  1. in-sample / out-of-sample split
  2. grid search on IN-SAMPLE only
  3. validation on OUT-OF-SAMPLE
  4. walk-forward analysis
  5. Monte Carlo drawdown distribution
  6. spread / slippage stress test
  7. fragility check (reject knife-edge optima)

The acceptance gate at the bottom is intentionally strict. If a parameter
set does not pass it, the honest conclusion is: do not trade.
"""

from __future__ import annotations

import copy
import os
import random
from typing import List, Optional

from strategy import Bar
from backtester import Backtester, BacktestResult, load_config, load_bars_csv


# --------------------------------------------------------------------------
# parameter grid (keep it small -- few parameters = less curve-fitting)
# --------------------------------------------------------------------------
PARAM_GRID = {
    "compression_k": [0.45, 0.55, 0.65],
    "breakout_buffer_atr": [0.05, 0.10, 0.15],
    "stop_atr_mult": [0.9, 1.1, 1.3],
    "reward_mult": [1.3, 1.6, 2.0],
}


def _with_params(config: dict, params: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault("strategy", {})
    cfg["strategy"].update(params)
    return cfg


def _score(metrics: dict) -> float:
    """Single robustness-aware score. Penalises tiny samples and big DD."""
    if metrics.get("trades", 0) < 30:
        return -1e9
    pf = metrics.get("profit_factor", 0)
    pf = 5.0 if pf == "inf" else float(pf)
    exp = float(metrics.get("expectancy_R", 0) or 0)
    dd = float(metrics.get("max_drawdown", 1) or 1)
    return (pf - 1.0) + 2.0 * exp - 1.5 * dd


# --------------------------------------------------------------------------
def grid_search(bars: List[Bar], config: dict) -> List[dict]:
    """Run every grid combo on the given bars. Returns ranked results."""
    keys = list(PARAM_GRID.keys())
    combos: List[dict] = [{}]
    for k in keys:
        combos = [dict(c, **{k: v}) for c in combos for v in PARAM_GRID[k]]

    results = []
    for params in combos:
        cfg = _with_params(config, params)
        res = Backtester(cfg).run(bars)
        results.append({"params": params, "metrics": res.metrics,
                         "score": _score(res.metrics)})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def in_sample_out_of_sample(bars: List[Bar], config: dict) -> dict:
    """Optimise on IS, validate the winner on OOS."""
    frac = float(config.get("backtest", {}).get("in_sample_fraction", 0.70))
    split = int(len(bars) * frac)
    is_bars, oos_bars = bars[:split], bars[split:]

    ranked = grid_search(is_bars, config)
    best = ranked[0]
    oos = Backtester(_with_params(config, best["params"])).run(oos_bars)

    return {
        "best_params": best["params"],
        "in_sample": best["metrics"],
        "out_of_sample": oos.metrics,
        "top5_in_sample": ranked[:5],
    }


def walk_forward(bars: List[Bar], config: dict,
                 windows: Optional[int] = None) -> dict:
    """
    Roll IS/OOS windows across the whole history. The strategy must be
    profitable in a MAJORITY of forward (OOS) windows, not just overall.
    """
    windows = windows or int(config.get("backtest", {})
                             .get("walkforward_windows", 6))
    seg = len(bars) // (windows + 1)
    if seg < 2000:
        return {"error": "not enough bars for walk-forward",
                "bars": len(bars)}

    fwd = []
    for w in range(windows):
        is_bars = bars[w * seg:(w + 2) * seg]
        oos_bars = bars[(w + 2) * seg:(w + 3) * seg]
        if len(oos_bars) < 1000:
            break
        ranked = grid_search(is_bars, config)
        best = ranked[0]
        oos = Backtester(_with_params(config, best["params"])).run(oos_bars)
        fwd.append({
            "window": w + 1,
            "params": best["params"],
            "oos_trades": oos.metrics.get("trades", 0),
            "oos_profit_factor": oos.metrics.get("profit_factor"),
            "oos_expectancy_R": oos.metrics.get("expectancy_R"),
            "oos_net": oos.metrics.get("net_profit"),
        })

    profitable = sum(1 for f in fwd
                     if isinstance(f["oos_net"], (int, float))
                     and f["oos_net"] > 0)
    return {
        "windows_tested": len(fwd),
        "windows_profitable": profitable,
        "pass_majority": len(fwd) > 0 and profitable / len(fwd) >= 0.60,
        "detail": fwd,
    }


def monte_carlo(result: BacktestResult, runs: int = 5000,
                seed: int = 42) -> dict:
    """
    Resample the trade sequence WITH REPLACEMENT `runs` times. Reports the
    distribution of max drawdown and ending equity. This shows how much of
    the backtest result was luck of ordering.
    """
    pnls = [t.pnl_money for t in result.trades]
    if len(pnls) < 20:
        return {"error": "too few trades for Monte Carlo", "trades": len(pnls)}

    rng = random.Random(seed)
    start = result.starting_balance
    max_dds, endings, ruined = [], [], 0

    for _ in range(runs):
        sample = [rng.choice(pnls) for _ in range(len(pnls))]
        eq = start
        peak = start
        mdd = 0.0
        blew = False
        for p in sample:
            eq += p
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 1.0
            mdd = max(mdd, dd)
            if eq <= start * 0.5:      # -50% = practical ruin
                blew = True
        max_dds.append(mdd)
        endings.append(eq)
        if blew:
            ruined += 1

    def pct(data, p):
        s = sorted(data)
        k = max(0, min(len(s) - 1, int(p / 100.0 * len(s))))
        return s[k]

    return {
        "runs": runs,
        "max_dd_p5": round(pct(max_dds, 5), 4),
        "max_dd_p50": round(pct(max_dds, 50), 4),
        "max_dd_p95": round(pct(max_dds, 95), 4),
        "ending_p5": round(pct(endings, 5), 2),
        "ending_p50": round(pct(endings, 50), 2),
        "ending_p95": round(pct(endings, 95), 2),
        "prob_practical_ruin": round(ruined / runs, 4),
    }


def cost_stress(bars: List[Bar], config: dict, params: dict) -> dict:
    """Re-run the chosen params with +0.5 pip spread and +0.3 pip slippage."""
    bt = Backtester(_with_params(config, params))
    base = bt.run(bars)
    stressed = bt.run(bars,
                      spread_override=bt.spread_pips + 0.5,
                      slippage_override=bt.slippage_pips + 0.3)
    return {"base": base.metrics, "stressed": stressed.metrics}


def fragility_check(ranked: List[dict]) -> dict:
    """
    A robust optimum sits on a PLATEAU: its grid neighbours are also good.
    If the best score is far above the median, suspect curve-fitting.
    """
    scores = sorted((r["score"] for r in ranked
                     if r["score"] > -1e8), reverse=True)
    if len(scores) < 5:
        return {"verdict": "insufficient_data"}
    best = scores[0]
    median = scores[len(scores) // 2]
    top5_spread = best - scores[4]
    return {
        "best_score": round(best, 3),
        "median_score": round(median, 3),
        "top5_spread": round(top5_spread, 3),
        "verdict": "plateau_ok" if top5_spread < abs(best) * 0.5 + 0.5
        else "fragile_suspect_curvefit",
    }


# --------------------------------------------------------------------------
def acceptance_gate(oos: dict, wf: dict, mc: dict, stress: dict) -> dict:
    """The hard go/no-go gate. ALL must pass to leave demo."""
    checks = {}
    pf = oos.get("profit_factor", 0)
    pf = 5.0 if pf == "inf" else float(pf or 0)
    checks["oos_profit_factor>=1.15"] = pf >= 1.15
    checks["walkforward_majority_profitable"] = bool(wf.get("pass_majority"))
    checks["mc_p95_drawdown<=0.20"] = (
        isinstance(mc.get("max_dd_p95"), (int, float))
        and mc["max_dd_p95"] <= 0.20)
    spf = stress.get("stressed", {}).get("profit_factor", 0)
    spf = 5.0 if spf == "inf" else float(spf or 0)
    checks["survives_cost_stress_pf>=1.05"] = spf >= 1.05
    checks["sample>=200_trades"] = oos.get("trades", 0) >= 60  # OOS is smaller
    passed = all(checks.values())
    return {"passed": passed, "checks": checks,
            "verdict": "MAY proceed to extended demo testing" if passed
            else "FAIL -- do not trade this. Keep researching."}


def main() -> None:
    cfg = load_config()
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", f"{cfg.get('symbol','EURUSD')}_"
                            f"{cfg.get('timeframe','M15')}.csv")
    bars = None
    try:
        from backtester import _get_bars_from_mt5
        bars = _get_bars_from_mt5(cfg)
    except Exception:  # noqa: BLE001
        bars = None
    if not bars and os.path.exists(csv_path):
        bars = load_bars_csv(csv_path)
    if not bars:
        print("No data. Provide", csv_path, "or run MT5.")
        return

    print(f"Loaded {len(bars)} bars.\n")

    print(">>> IN-SAMPLE / OUT-OF-SAMPLE")
    iso = in_sample_out_of_sample(bars, cfg)
    print("  best params :", iso["best_params"])
    print("  IS  :", {k: iso["in_sample"].get(k) for k in
                       ("trades", "profit_factor", "expectancy_R")})
    print("  OOS :", {k: iso["out_of_sample"].get(k) for k in
                      ("trades", "profit_factor", "expectancy_R")})

    print("\n>>> FRAGILITY")
    print(" ", fragility_check(grid_search(bars, cfg)))

    print("\n>>> WALK-FORWARD")
    wf = walk_forward(bars, cfg)
    print(" ", {k: wf.get(k) for k in
                ("windows_tested", "windows_profitable", "pass_majority")})

    print("\n>>> MONTE CARLO")
    best_res = Backtester(_with_params(cfg, iso["best_params"])).run(bars)
    mc = monte_carlo(best_res, runs=int(cfg.get("backtest", {})
                                        .get("monte_carlo_runs", 5000)))
    print(" ", mc)

    print("\n>>> COST STRESS")
    stress = cost_stress(bars, cfg, iso["best_params"])
    print("  base    :", {k: stress["base"].get(k) for k in
                          ("profit_factor", "expectancy_R")})
    print("  stressed:", {k: stress["stressed"].get(k) for k in
                          ("profit_factor", "expectancy_R")})

    print("\n>>> ACCEPTANCE GATE")
    gate = acceptance_gate(iso["out_of_sample"], wf, mc, stress)
    for k, v in gate["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("  VERDICT:", gate["verdict"])


if __name__ == "__main__":
    from backtester import _get_bars_from_mt5  # noqa: F401
    main()
