"""Transient market-impact overlay (Obizhaeva-Wang style).

A replay cannot show how the market would have reacted to our orders, so each
of our fills pushes our later prices against us, and the push decays with a
half-life. The overlay changes our prices, not which resting orders fill.
"""
import numpy as np


class TransientImpact:
    """Obizhaeva-Wang style: each fill pushes our later prices against us; the push decays."""

    def __init__(self, lam_bps_per_share, half_life_s):
        self.lam = lam_bps_per_share
        self.decay = np.log(2) / half_life_s
        self.level_bps, self.t_last = 0.0, None

    def current_bps(self, t):
        if self.t_last is None:
            return 0.0
        return self.level_bps * np.exp(-self.decay * (t - self.t_last))

    def add_fill(self, t, qty):
        self.level_bps = self.current_bps(t) + self.lam * qty
        self.t_last = t


def calibrate_lambda(Y, sigma_daily_bps, qty, daily_volume, schedule, slice_s, half_life_s):
    """lam such that executing `schedule` (one fill per slice) peaks at the
    square-root-law push Y * sigma_daily * sqrt(Q / V_daily) bps."""
    if Y <= 0 or qty <= 0:
        return 0.0
    target = Y * sigma_daily_bps * np.sqrt(qty / daily_volume)
    decay = np.exp(-np.log(2) / half_life_s * slice_s)
    level = peak = 0.0
    for q in schedule:                    # push path with lam = 1; linear in lam
        level = level * decay + q
        peak = max(peak, level)
    return target / peak if peak > 0 else 0.0
