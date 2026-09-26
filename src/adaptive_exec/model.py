"""Ridge (baseline) and XGBoost (main) models of the forward mid move.

One pair of models per label horizon (10, 30 and 60 s by default); models are
keyed like "xgboost_30s". Targets are clipped at percentiles of the TRAINING
targets, the ridge scaler and imputation medians come from training rows only,
and XGBoost early-stops on the validation rows. The final choice of model and
horizon is made later, on validation data, by execution cost, not by these
prediction metrics.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

MODEL_NAMES = ("ridge", "xgboost")


def label_column(horizon_s: int) -> str:
    return f"y_bps_{horizon_s}"


def model_key(name: str, horizon_s: int) -> str:
    return f"{name}_{horizon_s}s"


def key_horizon(key: str) -> int:
    return int(key.rsplit("_", 1)[1].rstrip("s"))


def usable(df: pd.DataFrame, ycol: str) -> pd.Series:
    """Rows with a label; XGBoost handles missing features, ridge gets them imputed."""
    return df[ycol].notna() & df[FEATURE_COLUMNS].notna().any(axis=1)


class ReturnModels:
    def __init__(self, horizons=(30,), ridge_alpha=10.0, clip_pct=(0.5, 99.5),
                 xgb_params: dict | None = None):
        self.horizons = tuple(int(h) for h in horizons)
        self.ridge_alpha = ridge_alpha
        self.clip_pct = tuple(clip_pct)
        self.xgb_params = xgb_params or {}
        self.models: dict = {}
        self.clip: dict = {}
        self.best_iteration: dict = {}
        self.fill_values = None     # training medians, used to impute NaNs for ridge

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "ReturnModels":
        self.fill_values = train[FEATURE_COLUMNS].median()
        for h in self.horizons:
            ycol = label_column(h)
            tr, va = train[usable(train, ycol)], val[usable(val, ycol)]
            lo, hi = np.percentile(tr[ycol], self.clip_pct)
            self.clip[h] = (float(lo), float(hi))
            y_tr, y_va = tr[ycol].clip(lo, hi), va[ycol].clip(lo, hi)
            X_tr, X_va = tr[FEATURE_COLUMNS], va[FEATURE_COLUMNS]

            self.models[model_key("ridge", h)] = make_pipeline(
                StandardScaler(), Ridge(alpha=self.ridge_alpha)).fit(X_tr.fillna(self.fill_values), y_tr)
            params = dict(n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=50, early_stopping_rounds=100,
                          eval_metric="rmse", n_jobs=-1, random_state=0)
            params.update(self.xgb_params)
            booster = xgb.XGBRegressor(**params)
            booster.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            self.models[model_key("xgboost", h)] = booster
            self.best_iteration[model_key("xgboost", h)] = int(booster.best_iteration)
        return self

    def keys(self) -> list[str]:
        return list(self.models)

    def predict(self, key: str, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLUMNS]
        if key.startswith("ridge"):
            X = X.fillna(self.fill_values)
        return self.models[key].predict(X)

    def feature_importance(self) -> pd.DataFrame:
        cols = {}
        for h in self.horizons:
            gain = self.models[model_key("xgboost", h)].get_booster().get_score(importance_type="gain")
            g = pd.Series(gain).reindex(FEATURE_COLUMNS).fillna(0.0)
            cols[f"xgboost_{h}s_gain_share"] = g / g.sum() if g.sum() > 0 else g
            cols[f"ridge_{h}s_std_coef"] = pd.Series(self.models[model_key("ridge", h)][-1].coef_,
                                                     index=FEATURE_COLUMNS)
        return pd.DataFrame(cols)

    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "models.pkl", "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(directory) -> "ReturnModels":
        with open(Path(directory) / "models.pkl", "rb") as fh:
            return pickle.load(fh)


def prediction_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Out-of-sample R^2, correlation, and sign hit rate on non-zero moves."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    nz = y != 0
    return {
        "n": int(ok.sum()),
        "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(y) > 1 else float("nan"),
        "corr": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and p.std() > 0 else float("nan"),
        "hit_rate_nonzero": float((np.sign(p[nz]) == np.sign(y[nz])).mean()) if nz.any() else float("nan"),
        "share_zero_moves": float(1 - nz.mean()) if len(y) else float("nan"),
    }


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
