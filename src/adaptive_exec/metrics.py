"""Implementation shortfall, its five-part decomposition, and block-bootstrap CIs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def implementation_shortfall_bps(side, arrival_mid, fills, fees):
    """fills: [(price, qty), ...] after the impact adjustment; fees: net dollars (rebates negative)."""
    qty = sum(q for _, q in fills)
    avg_px = sum(p * q for p, q in fills) / qty
    price_cost = side * (avg_px - arrival_mid) / arrival_mid * 1e4
    fee_cost = fees / (arrival_mid * qty) * 1e4
    return price_cost + fee_cost


@dataclass
class Fill:
    t: float           # seconds since the parent order started
    px: float          # raw price from the replay
    px_adj: float      # after the impact overlay
    mid: float         # historical mid at the time of the fill
    qty: int
    passive: bool
    fee: float         # dollars; rebates negative


def decompose(side, arrival_mid, parent_qty, fills: list[Fill],
              unfilled_qty=0, final_mid=np.nan, final_half_spread=np.nan) -> dict:
    """Five parts, in bps of arrival notional for the whole parent order, that add up to IS.

    spread = side * (p - m) / A      impact = side * (p' - p) / A
    drift  = side * (m - A) / A      fees   = fees / A
    opportunity: unfilled shares charged at final mid + half spread, vs arrival.
    """
    A, Q = arrival_mid, parent_qty
    scale = 1e4 / (A * Q)
    spread = sum(side * (f.px - f.mid) * f.qty for f in fills) * scale
    impact = sum(side * (f.px_adj - f.px) * f.qty for f in fills) * scale
    drift = sum(side * (f.mid - A) * f.qty for f in fills) * scale
    fees = sum(f.fee for f in fills) * scale
    opp = 0.0
    if unfilled_qty > 0:
        opp = side * (final_mid + side * final_half_spread - A) * unfilled_qty * scale
    return {"is_bps": spread + impact + drift + fees + opp, "spread_bps": spread,
            "drift_bps": drift, "impact_bps": impact, "fees_bps": fees, "opportunity_bps": opp}


def block_bootstrap_ci(values, blocks, n_boot=2000, alpha=0.05, seed=0):
    """CI for the mean of `values`, resampling whole blocks with replacement."""
    df = pd.DataFrame({"v": np.asarray(values, float), "b": np.asarray(blocks)})
    g = df.groupby("b")["v"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(df["v"].mean()), float(lo), float(hi)
