import numpy as np
import pytest

from adaptive_exec.impact import TransientImpact, calibrate_lambda


def test_impact_halves_after_one_half_life():
    imp = TransientImpact(lam_bps_per_share=0.001, half_life_s=300)
    imp.add_fill(t=0.0, qty=1000)                  # push = 1 bp
    assert imp.current_bps(300.0) == pytest.approx(0.5)


def test_calibration_hits_the_square_root_target():
    sched = np.full(60, 100)
    lam = calibrate_lambda(0.5, 150.0, 6000, 6_000_000, sched, 30, 300)
    imp, peak = TransientImpact(lam, 300), 0.0
    for k, q in enumerate(sched):
        imp.add_fill(30.0 * k, q)
        peak = max(peak, imp.current_bps(30.0 * k))
    assert peak == pytest.approx(0.5 * 150.0 * np.sqrt(6000 / 6_000_000))
    assert calibrate_lambda(0.0, 150.0, 6000, 6_000_000, sched, 30, 300) == 0.0
