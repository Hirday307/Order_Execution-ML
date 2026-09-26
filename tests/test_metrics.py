import numpy as np
import pytest

from adaptive_exec.metrics import Fill, block_bootstrap_ci, decompose, implementation_shortfall_bps


def test_paying_up_is_a_positive_cost_for_both_sides():
    assert implementation_shortfall_bps(+1, 100.0, [(100.05, 1000)], fees=0.0) == pytest.approx(5.0)
    assert implementation_shortfall_bps(-1, 100.0, [(99.95, 1000)], fees=0.0) == pytest.approx(5.0)


@pytest.mark.parametrize("side", [+1, -1])
def test_decomposition_adds_up_to_implementation_shortfall(side):
    A = 100.0
    fills = [Fill(0, 100.01 if side == 1 else 99.99, 100.012 if side == 1 else 99.988, 100.00, 400, False, 1.2),
             Fill(60, 100.03 if side == 1 else 99.97, 100.03 if side == 1 else 99.97, 100.04, 300, True, -0.6)]
    unfilled, final_mid, half = 300, 100.10, 0.01
    d = decompose(side, A, 1000, fills, unfilled, final_mid, half)
    parts = d["spread_bps"] + d["drift_bps"] + d["impact_bps"] + d["fees_bps"] + d["opportunity_bps"]
    assert d["is_bps"] == pytest.approx(parts)
    # Same number from the plain IS formula, with the opportunity cost as a pseudo-fill
    all_fills = [(f.px_adj, f.qty) for f in fills] + [(final_mid + side * half, unfilled)]
    assert d["is_bps"] == pytest.approx(implementation_shortfall_bps(side, A, all_fills, fees=0.6))
    # A resting fill below mid earns the spread on a buy
    assert decompose(1, A, 300, [fills[1]])["spread_bps"] < 0 if side == 1 else True


def test_block_bootstrap_ci_contains_the_mean_and_widens_with_correlation():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(400)
    m, lo, hi = block_bootstrap_ci(iid, np.arange(400))
    assert lo < m < hi
    shocks = np.repeat(rng.standard_normal(20), 20) + 0.1 * rng.standard_normal(400)
    _, lo_b, hi_b = block_bootstrap_ci(shocks, np.repeat(np.arange(20), 20))
    _, lo_i, hi_i = block_bootstrap_ci(shocks, np.arange(400))
    assert (hi_b - lo_b) > 2 * (hi_i - lo_i)
