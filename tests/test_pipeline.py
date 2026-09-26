from pathlib import Path

import pytest

from adaptive_exec.pipeline import make_units, prior_dates

SPLIT = {"mode": "intraday", "train": ["09:45", "12:30"], "validate": ["12:40", "14:00"],
         "test": ["14:10", "16:00"]}


def fake_processed(root: Path, symbols, days):
    for s in symbols:
        for d in days:
            p = root / s / d
            p.mkdir(parents=True)
            (p / "grid.parquet").touch()


def cfg_for(tmp_path, symbols, split):
    return {"processed_dir": str(tmp_path), "symbols": symbols, "split": split,
            "label": {"horizon_s": 30, "horizons": [10, 30, 60]}}


def test_per_stock_and_leave_one_stock_out_units(tmp_path):
    syms = ["AAA", "BBB", "CCC"]
    fake_processed(tmp_path, syms, ["2012-06-21"])
    units = make_units(cfg_for(tmp_path, syms, {**SPLIT, "leave_one_stock_out": True}))
    per = [u for u in units if u.kind == "per_stock"]
    loso = [u for u in units if u.kind == "loso"]
    assert len(per) == 3 and len(loso) == 3
    for u in loso:
        held_out = u.test.symbol
        assert held_out not in {p.symbol for p in u.train + u.validate}
        assert {p.symbol for p in u.train} == set(syms) - {held_out}
        # the held-out stock's scaling comes from its OWN earlier training window
        assert [p.window for p in u.norm[held_out]] == [u.train[0].window]
        assert all(p.window.end <= u.validate[0].window.start for p in u.train)
        assert u.validate[0].window.end <= u.test.window.start


def test_walk_forward_units_and_prior_dates(tmp_path):
    days = [f"2026-03-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18)]
    fake_processed(tmp_path, ["MSFT"], days)
    split = {"mode": "walk_forward", "session": ["09:45", "16:00"], "train_days": 10}
    units = make_units(cfg_for(tmp_path, ["MSFT"], split))
    assert [u.test.date for u in units] == ["2026-03-17", "2026-03-18"]
    u = units[-1]
    assert prior_dates(u, u.validate[0]) == days[1:11]          # curve for tuning: before d-1
    assert prior_dates(u, u.test) == days[1:12]                 # curve for testing: before d
    assert u.test.date not in prior_dates(u, u.test)


def test_missing_data_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_units(cfg_for(tmp_path, ["NONE"], SPLIT))
