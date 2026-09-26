import numpy as np
import pandas as pd

from adaptive_exec.schedules import (BUCKETS, almgren_chriss, curve_from_days, flat_curve,
                                     slice_weights, twap, vwap)


def test_schedules_trade_the_whole_order():
    for s in (twap(10_000, 60), vwap(10_000, np.linspace(2, 1, 60)), almgren_chriss(10_000, 60, 2.0)):
        assert s.sum() == 10_000
        assert (s >= 0).all()


def test_almgren_chriss_is_front_loaded_and_reduces_to_twap():
    assert almgren_chriss(10_000, 60, 2.0)[0] > almgren_chriss(10_000, 60, 2.0)[-1]
    assert (almgren_chriss(10_000, 60, 1e-9) == twap(10_000, 60)).all()


def test_volume_curve_covers_the_session_and_maps_to_slices():
    assert len(BUCKETS) == 78 and BUCKETS[0] == "09:30" and BUCKETS[-1] == "15:55"
    starts = pd.date_range("2012-06-21 14:10", periods=60, freq="30s")
    w = slice_weights(flat_curve(), starts, 30)
    assert np.isclose(w.sum(), 30 / 390)                     # 30 minutes of a 390-minute day


def test_curve_from_days_averages_shares():
    a = pd.Series(1.0, index=BUCKETS)
    b = pd.Series(0.0, index=BUCKETS)
    b.iloc[0] = 10.0
    c = curve_from_days([a, b])
    assert np.isclose(c.sum(), 1.0) and c.iloc[0] > c.iloc[1]
