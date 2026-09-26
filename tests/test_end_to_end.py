"""Every script, in order, on a tiny synthetic project (a few seconds of CPU)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from adaptive_exec.synthetic import write_lobster_day

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    d = tmp_path_factory.mktemp("proj")
    for i, s in enumerate(["AAA", "BBB"]):
        write_lobster_day(d / "raw", s, "2012-06-21", seed=40 + i, rate_per_s=1.5)
    cfg = yaml.safe_load((ROOT / "configs" / "synthetic.yaml").read_text())
    cfg.update(stage="e2e", raw_dir=str(d / "raw"), processed_dir=str(d / "processed"),
               models_dir=str(d / "models"), results_dir=str(d / "results"), symbols=["AAA", "BBB"],
               n_jobs=1)
    cfg["label"]["horizons"] = [10, 30]
    cfg["orders"]["every_min"] = 30
    cfg["model"]["xgboost"] = {"n_estimators": 50}
    cfg["tuning"].update(ac_kappa_T=[1], heuristic_theta=[0.5], alpha=[0.5], ml_m=[0.5],
                         models=["xgboost"], horizons=[10, 30])
    cfg["bootstrap"]["n_boot"] = 200
    path = d / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return d, path


def test_scripts_run_in_order_and_protect_the_test_period(project):
    d, cfg = project
    for script, extra in (("build_dataset.py", []), ("train.py", []),
                          ("backtest.py", ["--validate-only"])):
        r = run(script, "--config", str(cfg), *extra)
        assert r.returncode == 0, r.stderr
    assert not (d / "results" / "test_orders.csv").exists()          # validate-only never scores

    assert run("backtest.py", "--config", str(cfg)).returncode == 0
    orders = pd.read_csv(d / "results" / "test_orders.csv")
    assert set(orders["kind"]) == {"per_stock", "loso"}
    assert orders.groupby(["unit", "strategy", "impact", "fill_mode"]).size().nunique() == 1

    again = run("backtest.py", "--config", str(cfg))
    assert again.returncode != 0 and "already been scored" in again.stderr
    assert run("backtest.py", "--config", str(cfg), "--rescore").returncode != 0   # needs a reason
    assert run("backtest.py", "--config", str(cfg), "--rescore", "--reason", "e2e test").returncode == 0
    log = pd.read_csv(d / "results" / "scoring_log.csv")
    assert log["rescore"].tolist() == [False, True]

    readme = d / "README.md"
    readme.write_text("x\n<!-- BEGIN RESULTS: e2e -->\nold\n<!-- END RESULTS: e2e -->\ny\n")
    r = run("report.py", "--config", str(cfg), "--readme", str(readme))
    assert r.returncode == 0, r.stderr
    text = readme.read_text()
    assert "old" not in text and "| VWAP |" in text and text.endswith("y\n")
    summary = (d / "results" / "summary.md").read_text()
    for section in ("Success criterion", "Leave-one-stock-out", "Validation", "scored 2 time(s)"):
        assert section in summary

    r = run("inspect_order.py", "--config", str(cfg), "--start", "12:40")
    assert r.returncode == 0, r.stderr
    assert "plain formula" in r.stdout
