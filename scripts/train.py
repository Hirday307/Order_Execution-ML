"""Fit ridge and XGBoost for every label horizon and evaluation unit.

    python scripts/train.py --config configs/stage1_lobster.yaml

Prediction metrics are reported on the training and validation rows only. The
test period is never touched here.
"""
import argparse
from pathlib import Path

import pandas as pd

from adaptive_exec.model import ReturnModels, key_horizon, label_column, prediction_metrics, save_json
from adaptive_exec.pipeline import fit_symbol_normalizers, load_config, make_units, training_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = Path(cfg["results_dir"])
    m = cfg["model"]
    rows, importances = [], []

    for unit in make_units(cfg):
        norms = fit_symbol_normalizers(cfg, unit)
        train, val = training_frames(cfg, unit, norms)
        models = ReturnModels(cfg["label"]["horizons"], m["ridge_alpha"], m["clip_pct"],
                              m.get("xgboost")).fit(train, val)
        out = Path(cfg["models_dir"]) / unit.name
        models.save(out)
        save_json(norms, out / "normalizers.json")

        for key in models.keys():
            ycol = label_column(key_horizon(key))
            for split, df in (("train", train), ("validate", val)):
                pm = prediction_metrics(df[ycol].to_numpy(), models.predict(key, df))
                rows.append({"unit": unit.name, "kind": unit.kind, "model": key.rsplit("_", 1)[0],
                             "horizon_s": key_horizon(key), "split": split, **pm})
        imp = models.feature_importance().rename_axis("feature").reset_index()
        importances.append(imp.assign(unit=unit.name, kind=unit.kind))

        v = pd.DataFrame([r for r in rows if r["unit"] == unit.name and r["split"] == "validate"])
        line = " | ".join(f"{r.model} {r.horizon_s}s corr {r.corr:+.3f}"
                          for r in v.itertuples() if r.horizon_s == cfg["label"]["horizon_s"])
        print(f"[train] {unit.name} ({unit.kind}): {len(train):,} train rows, validation {line}")

    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results / "prediction_metrics.csv", index=False)
    pd.concat(importances).to_csv(results / "feature_importance.csv", index=False)


if __name__ == "__main__":
    main()
