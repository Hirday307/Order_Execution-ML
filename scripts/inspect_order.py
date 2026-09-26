"""Trace one parent order decision by decision and fill by fill, to check it by hand.

    python scripts/inspect_order.py --config configs/stage1_lobster.yaml --unit MSFT_2012-06-21 \
        --strategy VWAP --start 12:40 --side buy --size 0.05

Uses a validation period by default. Looking at a test order before the test
period has been scored would be peeking, so `--period test` requires the
scoring log to exist.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from adaptive_exec.experiment import (FAMILIES, load_unit_models, make_strategy, period_market,
                                      settings)
from adaptive_exec.metrics import implementation_shortfall_bps
from adaptive_exec.pipeline import load_config, load_events, main_kind, make_units
from adaptive_exec.simulator import MarketReplay, run_parent_order
from adaptive_exec.splits import at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--unit", help="unit name (default: the first main unit)")
    ap.add_argument("--period", choices=["validate", "test"], default="validate")
    ap.add_argument("--strategy", choices=FAMILIES, default="VWAP")
    ap.add_argument("--params", help="JSON strategy parameters (default: chosen on validation)")
    ap.add_argument("--start", help="HH:MM start of the parent order (default: the first order)")
    ap.add_argument("--side", choices=["buy", "sell"], default="buy")
    ap.add_argument("--size", type=float, default=0.05, help="share of expected volume")
    ap.add_argument("--impact", default=None, help="impact setting (default: the tuning setting)")
    ap.add_argument("--fills", default=None, help="fill mode (default: the tuning setting)")
    ap.add_argument("--csv", help="write the decision and fill tables to this path prefix")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(cfg["results_dir"])

    units = make_units(cfg)
    unit = next(u for u in units if u.name == args.unit) if args.unit else \
        next(u for u in units if u.kind == main_kind(cfg))
    if args.period == "test" and not (res / "scoring_log.csv").exists():
        raise SystemExit("the test period has not been scored yet; inspecting it now would be peeking")
    period = unit.validate[0] if args.period == "validate" else unit.test

    if args.params:
        params = json.loads(args.params)
    elif (res / "chosen_params.json").exists():
        params = json.loads((res / "chosen_params.json").read_text())[unit.name][args.strategy]
    elif args.strategy in ("TWAP", "VWAP"):
        params = {}
    else:
        raise SystemExit("no chosen_params.json yet: pass --params")
    strategy = make_strategy(args.strategy, params)

    models, norms = load_unit_models(cfg, unit)
    market, orders, weights = period_market(cfg, unit, period, models, norms)
    side = 1 if args.side == "buy" else -1
    pick = [i for i, o in enumerate(orders) if o.side == side and abs(o.size_target - args.size) < 1e-9
            and (args.start is None or o.start == at(period.date, args.start))]
    if not pick:
        raise SystemExit("no parent order matches; starts are every "
                         f"{cfg['orders']['every_min']} min from {period.window.start:%H:%M}")
    order = orders[pick[0]]
    t = cfg["tuning"]
    s = settings(cfg, args.impact or t["impact"], args.fills or t["fill_mode"])
    replay = MarketReplay(load_events(cfg, period.symbol, period.date), cfg["levels"])
    r = run_parent_order(replay, order, strategy, market, s, weights[pick[0]], trace=True)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
    print(f"{unit.name} | {period.symbol} {period.date} | {strategy.name} {strategy.params()} | "
          f"{args.side} {order.qty:,} shares ({args.size:.0%} of expected volume) from "
          f"{order.start:%H:%M:%S} for {order.horizon_s // 60} min | impact {args.impact or t['impact']} "
          f"(lambda {r['impact_lambda']:.3g} bps/share), {s.fill_mode} fills")
    print(f"arrival mid {r['arrival_mid']:.4f}\n")
    dec = pd.DataFrame(r["decisions"])
    print(dec.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    fills = pd.DataFrame([f.__dict__ for f in r["fills"]])
    print(f"\n{len(fills)} fills")
    if len(fills):
        print(fills.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # Independent check: the plain IS formula, with unfilled shares as a pseudo-fill
    pseudo = [(f.px_adj, f.qty) for f in r["fills"]]
    if r["unfilled"]:
        pseudo.append((r["end_mid"] + side * r["end_half_spread"], r["unfilled"]))
    fees = sum(f.fee for f in r["fills"])
    is_check = implementation_shortfall_bps(side, r["arrival_mid"], pseudo, fees)
    parts = ["spread_bps", "drift_bps", "impact_bps", "fees_bps", "opportunity_bps"]
    print("\ncost (bps): " + ", ".join(f"{p[:-4]} {r[p]:+.3f}" for p in parts) +
          f" | total IS {r['is_bps']:+.3f} | plain formula {is_check:+.3f}")
    print(f"filled {r['filled']:,} (passive {r['passive_share']:.1%}), unfilled {r['unfilled']:,}, "
          f"catch-ups {r['n_catchups']}, participation {r['participation']:.1%}")
    if args.csv:
        dec.to_csv(f"{args.csv}_decisions.csv", index=False)
        fills.to_csv(f"{args.csv}_fills.csv", index=False)


if __name__ == "__main__":
    main()
