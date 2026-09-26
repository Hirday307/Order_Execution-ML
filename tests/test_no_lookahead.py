import numpy as np
import pandas as pd

from adaptive_exec.features import FEATURE_COLUMNS, build_features, make_grid
from adaptive_exec.labels import forward_move_bps


def test_features_do_not_use_the_future(events):              # `events` is a synthetic fixture
    cut = events["ts"].iloc[len(events) // 2]
    full = build_features(events).set_index("ts")
    past = build_features(events[events["ts"] <= cut]).set_index("ts")
    rows = past.index[past.index <= cut]
    pd.testing.assert_frame_equal(full.loc[rows, FEATURE_COLUMNS], past.loc[rows, FEATURE_COLUMNS])


def test_features_ignore_an_event_at_the_same_second_but_later(events):
    """A grid row at g must not see events stamped after g, even within the same second."""
    grid = make_grid(events)
    g = grid[len(grid) // 2]
    f = build_features(events, grid=grid).set_index("ts")
    shifted = events.copy()
    after = shifted["ts"] > g
    shifted.loc[after, "size"] = shifted.loc[after, "size"] * 10
    shifted.loc[after, ["bid_sz_1", "ask_sz_1"]] = shifted.loc[after, ["bid_sz_1", "ask_sz_1"]] * 3
    f2 = build_features(shifted, grid=grid).set_index("ts")
    pd.testing.assert_frame_equal(f.loc[:g], f2.loc[:g])


def test_features_are_mostly_defined_after_warmup(events):
    f = build_features(events)
    late = f[f["minutes_since_open"] >= 6]
    assert late[FEATURE_COLUMNS].notna().all().all()


def test_label_is_the_forward_move_and_stops_at_the_data_end(events):
    grid = make_grid(events)
    y = forward_move_bps(events, grid, horizon_s=30)
    assert np.isnan(y[-30:]).all() and np.isfinite(y[:-31]).all()
    v = events[events["valid"]]
    i = 100
    now = v[v["ts"] <= grid[i]]["mid"].iloc[-1]
    ahead = v[v["ts"] <= grid[i] + pd.Timedelta(seconds=30)]["mid"].iloc[-1]
    assert y[i] == (ahead / now - 1) * 1e4
