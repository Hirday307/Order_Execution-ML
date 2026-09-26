"""Strategy candidates and per-period setup shared by the backtest, the order
inspector and the notebooks."""
import itertools
import json
from pathlib import Path

import pandas as pd

from .model import ReturnModels, model_key
from .pipeline import (Period, Unit, normalized_grid, prediction_lookup, reference_daily_volume,
                       symbol_sigma, volume_curve)
from .simulator import ExecSettings, MarketContext, expected_volume, make_parent_orders, order_weights
from .strategies import TWAP, VWAP, AlmgrenChriss, HeuristicAdaptive, MLAdaptive

FAMILIES = ["TWAP", "VWAP", "Almgren-Chriss", "Heuristic adaptive",
            "ML adaptive (tactic only)", "ML adaptive (tactic + speed)"]
METRICS = ["is_bps", "spread_bps", "drift_bps", "impact_bps", "fees_bps", "opportunity_bps",
           "passive_share", "n_catchups"]


def ml_keys(cfg) -> list[str]:
    t = cfg["tuning"]
    return [model_key(m, h) for m in t["models"] for h in t.get("horizons", cfg["label"]["horizons"])]


def candidates(cfg) -> list[tuple[str, object]]:
    """Every setting tried on validation. Within a family, candidates run from the most
    conservative to the most aggressive, and ties go to the first, most conservative one."""
    t = cfg["tuning"]
    m_values = sorted(t["ml_m"], reverse=True)
    out = [("TWAP", TWAP()), ("VWAP", VWAP())]
    out += [("Almgren-Chriss", AlmgrenChriss(k)) for k in sorted(t["ac_kappa_T"])]
    out += [("Heuristic adaptive", HeuristicAdaptive(th, a))
            for th, a in itertools.product(sorted(t["heuristic_theta"], reverse=True), sorted(t["alpha"]))]
    out += [("ML adaptive (tactic only)", MLAdaptive(k, m, 0.0))
            for k, m in itertools.product(ml_keys(cfg), m_values)]
    out += [("ML adaptive (tactic + speed)", MLAdaptive(k, m, a))
            for k, m, a in itertools.product(ml_keys(cfg), m_values, sorted(t["alpha"]))]
    return out


def make_strategy(family: str, params: dict):
    if family == "TWAP":
        return TWAP()
    if family == "VWAP":
        return VWAP()
    if family == "Almgren-Chriss":
        return AlmgrenChriss(params["kappa_T"])
    if family == "Heuristic adaptive":
        return HeuristicAdaptive(params["theta"], params["alpha"])
    if family in ("ML adaptive (tactic only)", "ML adaptive (tactic + speed)"):
        return MLAdaptive(params["model"], params["m"], params["alpha"])
    raise ValueError(f"unknown strategy family {family}")


def settings(cfg, impact: str, fill_mode: str) -> ExecSettings:
    imp = cfg["impact"][impact]
    return ExecSettings(
        taker_fee=cfg["fees"]["taker_per_share"], maker_rebate=cfg["fees"]["maker_rebate_per_share"],
        band=cfg["execution"]["band"], catchup_threshold=cfg["execution"]["catchup_threshold"],
        fill_mode=fill_mode, impact_Y=imp["Y"], impact_half_life_s=imp["half_life_s"])


def load_unit_models(cfg, unit: Unit) -> tuple[ReturnModels, dict]:
    d = Path(cfg["models_dir"]) / unit.name
    return ReturnModels.load(d), json.loads((d / "normalizers.json").read_text())


def period_market(cfg, unit: Unit, period: Period, models: ReturnModels, norms: dict):
    """Market context (curve, volatility, volume, predictions), parent orders and
    their slice weights for one period."""
    grid = normalized_grid(cfg, period, norms)
    preds = {k: prediction_lookup(grid, models.predict(k, grid)) for k in models.keys()}
    curve = volume_curve(cfg, unit, period)
    v_ref = reference_daily_volume(cfg, unit, period)
    market = MarketContext(curve=curve, sigma_daily_bps=symbol_sigma(cfg, unit, period.symbol),
                           daily_volume=v_ref, predictions=preds)
    o = cfg["orders"]
    orders = make_parent_orders(
        period.window.start, period.window.end,
        lambda s, h: expected_volume(curve, v_ref, s, h, o["slice_s"]),
        o["targets"], o["every_min"], o["horizon_min"], o["slice_s"])
    return market, orders, [order_weights(curve, od) for od in orders]


def aggregate(results: list[dict]) -> dict:
    df = pd.DataFrame(results)
    return {"n_orders": len(df), **{f"mean_{m}": float(df[m].mean()) for m in METRICS},
            "share_orders_unfilled": float((df["unfilled"] > 0).mean())}
