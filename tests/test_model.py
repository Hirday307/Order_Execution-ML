import numpy as np
import pandas as pd

from adaptive_exec.features import FEATURE_COLUMNS
from adaptive_exec.model import ReturnModels, key_horizon, model_key, prediction_metrics


def frame(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    signal = X["ofi_30s"].to_numpy()
    for h in (10, 30):
        X[f"y_bps_{h}"] = signal * (h / 10) + rng.standard_normal(n)
    X.loc[X.index[:5], "y_bps_30"] = np.nan                   # unlabeled rows are skipped
    return X


def test_one_ridge_and_one_xgboost_per_horizon_and_they_learn_a_signal():
    m = ReturnModels(horizons=(10, 30), xgb_params={"n_estimators": 200}).fit(frame(3000, 0), frame(1000, 1))
    assert sorted(m.keys()) == ["ridge_10s", "ridge_30s", "xgboost_10s", "xgboost_30s"]
    test = frame(1000, 2)
    for k in m.keys():
        pm = prediction_metrics(test[f"y_bps_{key_horizon(k)}"].to_numpy(), m.predict(k, test))
        assert pm["corr"] > 0.5, (k, pm)
    lo, hi = m.clip[30]
    assert lo < 0 < hi


def test_model_keys_round_trip():
    assert key_horizon(model_key("xgboost", 60)) == 60
