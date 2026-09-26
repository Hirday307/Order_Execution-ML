import importlib.util
import json
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "backtest", Path(__file__).resolve().parents[1] / "scripts" / "backtest.py")
backtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtest)


def test_choice_pools_validation_periods_and_breaks_ties_conservatively():
    rows = [
        # unit U validates on two stocks; m=2 and m=1 tie overall, m=0.5 is worse
        ("U", "S1", {"m": 2}, 1.0, 10), ("U", "S2", {"m": 2}, 3.0, 30),
        ("U", "S1", {"m": 1}, 0.0, 10), ("U", "S2", {"m": 1}, 10 / 3, 30),
        ("U", "S1", {"m": 0.5}, 0.0, 10), ("U", "S2", {"m": 0.5}, 4.0, 30),
    ]
    df = pd.DataFrame([{"unit": u, "symbol": s, "family": "F", "params": json.dumps(p),
                        "mean_is_bps": v, "n_orders": n} for u, s, p, v, n in rows])
    assert backtest.choose(df) == {"U": {"F": {"m": 2}}}
