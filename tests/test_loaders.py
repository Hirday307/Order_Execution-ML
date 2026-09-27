import numpy as np
import pandas as pd

from adaptive_exec.loaders import databento_to_events, lobster_to_events


def test_lobster_trade_sign_is_minus_the_resting_direction(raw_lobster):
    msg, book = raw_lobster
    ev = lobster_to_events(msg, book, "2012-06-21")
    tr = ev[ev["is_trade"]]
    assert len(tr) > 0
    execs = msg[msg["type"].isin([4, 5])]
    assert (tr["aggressor"].to_numpy() == -execs["direction"].to_numpy()).all()


def test_lobster_units_timestamps_and_dummy_levels():
    msg = pd.DataFrame({"time": [34200.5, 34201.0, 34202.0], "type": [1, 4, 7],
                        "order_id": [1, 2, 0], "size": [100, 50, 0],
                        "price": [300100, 300200, -1], "direction": [1, -1, -1]})
    book = pd.DataFrame([[300200, 200, 300100, 100],
                         [300200, 150, 300100, 100],
                         [9999999999, 0, -9999999999, 0]])
    ev = lobster_to_events(msg, book, "2012-06-21", levels=1)
    assert len(ev) == 2                                      # halt row dropped
    assert ev["ts"].dtype == "datetime64[ns]"
    assert ev["ts"].iloc[0] == pd.Timestamp("2012-06-21 09:30:00.5")
    assert ev["ask_px_1"].iloc[0] == 30.02 and ev["price"].iloc[1] == 30.02
    assert ev["aggressor"].iloc[1] == 1                      # resting sell hit -> buyer-initiated
    assert ev["valid"].all()


def test_lobster_masks_empty_levels():
    msg = pd.DataFrame({"time": [34200.0], "type": [1], "order_id": [1], "size": [100],
                        "price": [300100], "direction": [1]})
    book = pd.DataFrame([[9999999999, 0, 300100, 100]])
    ev = lobster_to_events(msg, book, "2012-06-21", levels=1)
    assert np.isnan(ev["ask_px_1"].iloc[0]) and not ev["valid"].iloc[0]


def test_databento_trade_side_is_the_aggressor():
    idx = pd.DatetimeIndex(["2026-03-02 14:30:01", "2026-03-02 14:30:02",
                            "2026-03-02 14:30:03"], tz="UTC")    # 09:30 New York
    df = pd.DataFrame({"action": ["A", "T", "T"], "side": ["B", "B", "A"],
                       "price": [30.00, 30.01, 30.00], "size": [100, 50, 70]}, index=idx)
    for j in range(10):
        df[f"bid_px_{j:02d}"] = 30.00 - j * 0.01
        df[f"ask_px_{j:02d}"] = 30.01 + j * 0.01
        df[f"bid_sz_{j:02d}"] = 100
        df[f"ask_sz_{j:02d}"] = 100
    ev = databento_to_events(df)
    assert ev["aggressor"].tolist() == [0, 1, -1]
    assert ev["order_side"].tolist() == [1, -1, 1]
    assert ev["ts"].iloc[0] == pd.Timestamp("2026-03-02 09:30:01")
    assert ev["is_trade"].tolist() == [False, True, True]


def test_databento_cancel_that_removes_a_traded_quantity_is_not_a_cancel():
    t = pd.Timestamp("2026-03-02 14:30:05", tz="UTC")
    idx = pd.DatetimeIndex([t, t + pd.Timedelta(microseconds=5), t + pd.Timedelta(microseconds=9),
                            t + pd.Timedelta(seconds=1)])
    df = pd.DataFrame({
        "ts_event": [t, t, t, t + pd.Timedelta(seconds=1)],
        "action": ["T", "C", "C", "C"], "side": ["A", "B", "B", "B"],
        "price": [30.00, 30.00, 29.99, 30.00], "size": [100, 100, 50, 70]}, index=idx)
    for j in range(10):
        df[f"bid_px_{j:02d}"] = 30.00 - j * 0.01
        df[f"ask_px_{j:02d}"] = 30.01 + j * 0.01
        df[f"bid_sz_{j:02d}"] = 100
        df[f"ask_sz_{j:02d}"] = 100
    ev = databento_to_events(df)
    # same timestamp, same price, resting side -> fill removal (8); others stay cancels (3)
    assert ev["type"].tolist() == [4, 8, 3, 3]


def test_lobster_hidden_executions_use_the_quote_rule():
    msg = pd.DataFrame({"time": [34200.0, 34201.0, 34202.0], "type": [5, 5, 5], "order_id": [0, 0, 0],
                        "size": [10, 20, 30], "price": [300050, 300100, 300000], "direction": [1, 1, 1]})
    book = pd.DataFrame([[300100, 100, 300000, 100]] * 3)          # bid 30.00, ask 30.01
    ev = lobster_to_events(msg, book, "2012-06-21", levels=1)
    assert ev["aggressor"].tolist() == [0, 1, -1]                  # at mid, at ask, at bid
    assert ev["is_trade"].all()
