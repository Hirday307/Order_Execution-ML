"""Tune every strategy on validation, then score the test periods ONCE.

    python scripts/backtest.py --config configs/stage1_lobster.yaml --validate-only   # tuning only
    python scripts/backtest.py --config configs/stage1_lobster.yaml                   # tuning + test

Writes to results/<stage>/:
  validation_tuning.csv  mean cost and its decomposition for every candidate setting
  chosen_params.json     the setting picked for each strategy family and unit
  test_orders.csv        every test parent order, strategy, impact setting and fill mode
  scoring_log.csv        one line each time a test period is scored

If test_orders.csv already exists the script refuses to score again: that test
period has been spent. `--rescore --reason "..."` overrides this, and the reason
goes into the scoring log and the report.
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from adaptive_exec.experiment import (FAMILIES, aggregate, candidates, load_unit_models,
                                     make_strategy, period_market, settings)
from adaptive_exec.model import save_json
from adaptive_exec.pipeline import load_config, load_events, make_units
from adaptive_exec.simulator import MarketReplay, run_parent_order


def _shared_key(period, family, params, market, extra=()):
    """Model-free strategies give identical results for every unit on the same period."""
    return (period, family, params, round(market.sigma_daily_bps, 9), market.daily_volume, extra)


# ---------------------------------------------------------------------- phases

def validate_group(cfg, key, items) -> list[dict]:
    """All validation runs that use the replay of one (symbol, date)."""
    symbol, date = key
    replay = MarketReplay(load_events(cfg, symbol, date), cfg["levels"])
    t = cfg["tuning"]
    s_tune = settings(cfg, t["impact"], t["fill_mode"])
    shared, rows = {}, []
    for unit, period in items:
        models, norms = load_unit_models(cfg, unit)
        market, orders, weights = period_market(cfg, unit, period, models, norms)
        for family, strat in candidates(cfg):
            params = json.dumps(strat.params(), sort_keys=True)
            ck = _shared_key(period, family, params, market) if strat.model is None else None
            if ck is not None and ck in shared:
                agg = shared[ck]
            else:
                agg = aggregate([run_parent_order(replay, o, strat, market, s_tune, w)
                                 for o, w in zip(orders, weights)])
                if ck is not None:
                    shared[ck] = agg
            rows.append({"unit": unit.name, "kind": unit.kind, "symbol": symbol, "date": date,
                         "family": family, "params": params, **agg})
    return rows


def choose(tuning: pd.DataFrame) -> dict:
    """Lowest mean IS per unit and family, pooled over the unit's validation periods."""
    df = tuning.assign(w=tuning["mean_is_bps"] * tuning["n_orders"])
    g = df.groupby(["unit", "family", "params"], sort=False).agg(n=("n_orders", "sum"), w=("w", "sum"))
    g = g.assign(pooled=g["w"] / g["n"]).reset_index()
    chosen = {}
    for (unit, family), sub in g.groupby(["unit", "family"], sort=False):
        best = sub.loc[sub["pooled"].idxmin()]            # first minimum = most conservative tie
        chosen.setdefault(unit, {})[family] = json.loads(best["params"])
    return chosen


def test_group(cfg, key, units, chosen) -> list[dict]:
    """All test runs that use the replay of one (symbol, date)."""
    symbol, date = key
    replay = MarketReplay(load_events(cfg, symbol, date), cfg["levels"])
    shared, rows = {}, []
    for unit in units:
        models, norms = load_unit_models(cfg, unit)
        market, orders, weights = period_market(cfg, unit, unit.test, models, norms)
        for family, impact, fill in itertools.product(FAMILIES, cfg["impact"], cfg["fill_modes"]):
            strat = make_strategy(family, chosen[unit.name][family])
            params = json.dumps(strat.params(), sort_keys=True)
            ck = (_shared_key(unit.test, family, params, market, (impact, fill))
                  if strat.model is None else None)
            if ck is not None and ck in shared:
                res = shared[ck]
            else:
                s = settings(cfg, impact, fill)
                res = [run_parent_order(replay, o, strat, market, s, w) for o, w in zip(orders, weights)]
                if ck is not None:
                    shared[ck] = res
            tags = {"unit": unit.name, "kind": unit.kind, "symbol": symbol, "date": date,
                    "impact": impact, "fill_mode": fill}
            rows += [{**tags, "order_id": i, **r} for i, r in enumerate(res)]
    return rows


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"git_commit": run("git", "rev-parse", "--short", "HEAD") or "none",
            "git_dirty": bool(run("git", "status", "--porcelain"))}


def parallel(cfg, fn, groups: dict, *extra) -> list[dict]:
    n_jobs = min(len(groups), cfg.get("n_jobs", -1) if cfg.get("n_jobs", -1) > 0 else len(groups))
    out = Parallel(n_jobs=n_jobs)(delayed(fn)(cfg, k, v, *extra) for k, v in groups.items())
    return [row for part in out for row in part]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--validate-only", action="store_true", help="tune on validation; do not touch test")
    ap.add_argument("--rescore", action="store_true", help="score a test period that was already used")
    ap.add_argument("--reason", default="", help="required with --rescore; goes into the scoring log")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["results_dir"])
    test_path = out / "test_orders.csv"
    if not args.validate_only:
        if test_path.exists() and not args.rescore:
            sys.exit(f"{test_path} exists: the test period has already been scored. Re-scoring spends "
                     "it again; use --rescore --reason '...' only if you will report that.")
        if args.rescore and not args.reason.strip():
            sys.exit("--rescore needs --reason, which is recorded in the scoring log and the report")

    units = make_units(cfg)
    t0 = time.time()
    val_groups = defaultdict(list)
    for u in units:
        for p in u.validate:
            val_groups[(p.symbol, p.date)].append((u, p))
    tuning = pd.DataFrame(parallel(cfg, validate_group, val_groups))
    out.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(out / "validation_tuning.csv", index=False)
    chosen = choose(tuning)
    save_json(chosen, out / "chosen_params.json")
    n_cand = tuning.groupby("unit").size().max()
    print(f"[tune] {len(units)} units, up to {n_cand} candidate runs each, {time.time() - t0:.0f}s")
    for u in units:
        c = chosen[u.name]
        print(f"[tune] {u.name}: AC {c['Almgren-Chriss']} | heuristic {c['Heuristic adaptive']} | "
              f"ML tactic {c['ML adaptive (tactic only)']} | ML tactic+speed {c['ML adaptive (tactic + speed)']}")
    if args.validate_only:
        print(f"[tune] wrote {out / 'validation_tuning.csv'}; test periods untouched")
        return

    t1 = time.time()
    test_groups = defaultdict(list)
    for u in units:
        test_groups[(u.test.symbol, u.test.date)].append(u)
    df = pd.DataFrame(parallel(cfg, test_group, test_groups, chosen))
    df["params"] = df["params"].map(lambda p: json.dumps(p, sort_keys=True))
    df.to_csv(test_path, index=False)
    log = out / "scoring_log.csv"
    entry = pd.DataFrame([{"scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           **git_state(), "rescore": args.rescore, "reason": args.reason,
                           "n_rows": len(df)}])
    entry.to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"[test] wrote {len(df):,} rows to {test_path} in {time.time() - t1:.0f}s; logged in {log}")


if __name__ == "__main__":
    main()
