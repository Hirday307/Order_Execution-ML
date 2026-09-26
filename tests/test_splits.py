import pytest

from adaptive_exec.splits import intraday_windows, walk_forward

SPLIT = {"train": ["09:45", "12:30"], "validate": ["12:40", "14:00"], "test": ["14:10", "16:00"]}


def test_intraday_windows_have_gaps_longer_than_the_label():
    w = intraday_windows("2012-06-21", SPLIT, label_horizon_s=30)
    assert w["train"].end < w["validate"].start < w["test"].start
    with pytest.raises(ValueError):
        intraday_windows("2012-06-21", SPLIT, label_horizon_s=600)


def test_walk_forward_trains_before_tuning_before_testing():
    days = [f"2026-03-{d:02d}" for d in range(2, 22)]
    folds = walk_forward(days, train_days=10)
    assert len(folds) == 9
    for f in folds:
        assert len(f.train_days) == 10
        assert max(f.train_days) < f.val_day < f.test_day
        assert days.index(f.test_day) - days.index(f.val_day) == 1
