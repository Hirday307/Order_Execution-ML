"""Strategies = a schedule (shares per slice) + a tactic rule (post or cross, speed).

Tactics returned by `decide`:
    "baseline"  post at the best price (catch-up crossing is handled by the simulator)
    "post"      same as baseline, used when the signal favours waiting
    "cross"     cross the spread for this slice
The speed multiplier scales the scheduled slice; the simulator keeps cumulative
shares inside the schedule band.
"""
from dataclasses import dataclass

import numpy as np

from . import schedules


@dataclass(frozen=True)
class DecisionContext:
    side: int
    pred_bps: float      # model's predicted 30 s mid move (NaN if no model)
    imb_l1: float
    spread_bps: float
    taker_bps: float
    rebate_bps: float


def ml_decision(side, pred_bps, spread_bps, taker_bps, rebate_bps, m, alpha):
    """Returns (tactic, speed multiplier) for this slice."""
    signed = side * pred_bps                       # > 0: price expected to move against us
    cross_cost = spread_bps + taker_bps + rebate_bps
    if signed > m * cross_cost:
        return "cross", 1 + alpha
    if signed < -m * cross_cost:
        return "post", 1 - alpha
    return "baseline", 1.0


def heuristic_decision(side, imb_l1, theta, alpha):
    signed = side * imb_l1                         # buying into a bid-heavy book: price likely to rise
    if signed > theta:
        return "cross", 1 + alpha
    if signed < -theta:
        return "post", 1 - alpha
    return "baseline", 1.0


def clamp_to_band(planned_cum, schedule_cum, total_qty, band=0.10):
    lo = max(0, schedule_cum - band * total_qty)
    hi = min(total_qty, schedule_cum + band * total_qty)
    return int(np.clip(planned_cum, lo, hi))


class Strategy:
    name = "base"
    model: str | None = None        # which model's predictions `decide` needs

    def schedule(self, qty: int, curve_weights: np.ndarray) -> np.ndarray:
        return schedules.vwap(qty, curve_weights)

    def decide(self, ctx: DecisionContext) -> tuple[str, float]:
        return "baseline", 1.0

    def params(self) -> dict:
        return {}


class TWAP(Strategy):
    name = "TWAP"

    def schedule(self, qty, curve_weights):
        return schedules.twap(qty, len(curve_weights))


class VWAP(Strategy):
    name = "VWAP"


class AlmgrenChriss(Strategy):
    name = "Almgren-Chriss"

    def __init__(self, kappa_T: float):
        self.kappa_T = kappa_T

    def schedule(self, qty, curve_weights):
        return schedules.almgren_chriss(qty, len(curve_weights), self.kappa_T)

    def params(self):
        return {"kappa_T": self.kappa_T}


class HeuristicAdaptive(Strategy):
    name = "Heuristic adaptive"

    def __init__(self, theta: float, alpha: float):
        self.theta, self.alpha = theta, alpha

    def decide(self, ctx):
        if not np.isfinite(ctx.imb_l1):
            return "baseline", 1.0
        return heuristic_decision(ctx.side, ctx.imb_l1, self.theta, self.alpha)

    def params(self):
        return {"theta": self.theta, "alpha": self.alpha}


class MLAdaptive(Strategy):
    """alpha = 0 is the 'tactic only' variant."""

    def __init__(self, model: str, m: float, alpha: float):
        self.model, self.m, self.alpha = model, m, alpha
        self.name = "ML adaptive (tactic only)" if alpha == 0 else "ML adaptive (tactic + speed)"

    def decide(self, ctx):
        if not np.isfinite(ctx.pred_bps):
            return "baseline", 1.0
        return ml_decision(ctx.side, ctx.pred_bps, ctx.spread_bps, ctx.taker_bps,
                           ctx.rebate_bps, self.m, self.alpha)

    def params(self):
        return {"model": self.model, "m": self.m, "alpha": self.alpha}
