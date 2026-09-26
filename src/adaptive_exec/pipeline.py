"""Config, processed-data IO, and evaluation units shared by the scripts.

A *unit* is one assignment of periods to train / validate / test:
  per_stock     Stage 1: one day per symbol, split by time of day.
  loso          Stage 1 extra check: train and tune on the other stocks' train and
                validation windows, test on this stock's test window.
  walk_forward  Stage 2: one fold per symbol and test day.
Each unit also names, per symbol, the periods its feature normalizers and
volatility estimate come from (`norm`). They are always earlier than, and never
part of, the validation and test periods.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .features import apply_normalizers, as_ns, fit_normalizers
from .schedules import curve_from_days, flat_curve, load_proxy_curve
from .splits import Window, at, intraday_windows, walk_forward


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_path"] = str(path)
    cfg["label"].setdefault("horizons", [cfg["label"]["horizon_s"]])
    return cfg


# ------------------------------------------------------------------ processed IO

def day_dir(cfg, symbol, date) -> Path:
    return Path(cfg["processed_dir"]) / symbol / date


def available_days(cfg, symbol) -> list[str]:
    d = Path(cfg["processed_dir"]) / symbol
    return sorted(p.name for p in d.iterdir() if (p / "grid.parquet").exists()) if d.exists() else []


def load_events(cfg, symbol, date) -> pd.DataFrame:
    return pd.read_parquet(day_dir(cfg, symbol, date) / "events.parquet")


@lru_cache(maxsize=32)
def _grid_cached(processed_dir: str, symbol: str, date: str) -> pd.DataFrame:
    return pd.read_parquet(Path(processed_dir) / symbol / date / "grid.parquet")


def load_grid(cfg, symbol, date) -> pd.DataFrame:
    return _grid_cached(str(cfg["processed_dir"]), symbol, date)


def load_meta(cfg, symbol, date) -> dict:
    return json.loads((day_dir(cfg, symbol, date) / "meta.json").read_text())


# ------------------------------------------------------------------------- units

@dataclass(frozen=True)
class Period:
    symbol: str
    date: str
    window: Window


@dataclass
class Unit:
    name: str
    kind: str                            # "per_stock", "loso" or "walk_forward"
    train: list[Period]
    validate: list[Period]
    test: Period
    norm: dict[str, list[Period]]        # symbol -> periods for normalizers and volatility

    def symbols(self) -> list[str]:
        return sorted(self.norm)


def make_units(cfg) -> list[Unit]:
    horizon = max(cfg["label"]["horizons"])
    split = cfg["split"]
    days = {s: available_days(cfg, s) for s in cfg["symbols"]}
    for s, d in days.items():
        if not d:
            raise FileNotFoundError(f"no processed data for {s}; run build_dataset.py first")
    units = []
    if split["mode"] == "intraday":
        for s in cfg["symbols"]:
            for date in days[s]:
                w = intraday_windows(date, split, horizon)
                units.append(Unit(f"{s}_{date}", "per_stock", [Period(s, date, w["train"])],
                                  [Period(s, date, w["validate"])], Period(s, date, w["test"]),
                                  {s: [Period(s, date, w["train"])]}))
        if split.get("leave_one_stock_out"):
            if len(cfg["symbols"]) < 2:
                raise ValueError("leave-one-stock-out needs at least two symbols")
            common = sorted(set.intersection(*(set(d) for d in days.values())))
            for date in common:
                w = intraday_windows(date, split, horizon)
                for s in cfg["symbols"]:
                    others = [o for o in cfg["symbols"] if o != s]
                    units.append(Unit(
                        f"LOSO_{s}_{date}", "loso",
                        [Period(o, date, w["train"]) for o in others],
                        [Period(o, date, w["validate"]) for o in others],
                        Period(s, date, w["test"]),
                        {x: [Period(x, date, w["train"])] for x in cfg["symbols"]}))
    elif split["mode"] == "walk_forward":
        a, b = split["session"]
        for s in cfg["symbols"]:
            for fold in walk_forward(days[s], split["train_days"]):
                train = [Period(s, d, Window(at(d, a), at(d, b))) for d in fold.train_days]
                units.append(Unit(
                    f"{s}_{fold.test_day}", "walk_forward", train,
                    [Period(s, fold.val_day, Window(at(fold.val_day, a), at(fold.val_day, b)))],
                    Period(s, fold.test_day, Window(at(fold.test_day, a), at(fold.test_day, b))),
                    {s: train}))
    else:
        raise ValueError(f"unknown split mode {split['mode']}")
    return units


def main_kind(cfg) -> str:
    return "per_stock" if cfg["split"]["mode"] == "intraday" else "walk_forward"


# ------------------------------------------------------------ frames and scaling

def period_grid(cfg, period: Period) -> pd.DataFrame:
    g = load_grid(cfg, period.symbol, period.date)
    return g[period.window.mask(g["ts"])].reset_index(drop=True)


def norm_frame(cfg, unit: Unit, symbol: str) -> pd.DataFrame:
    return pd.concat([period_grid(cfg, p) for p in unit.norm[symbol]], ignore_index=True)


def fit_symbol_normalizers(cfg, unit: Unit) -> dict[str, dict]:
    return {s: fit_normalizers(norm_frame(cfg, unit, s)) for s in unit.symbols()}


def normalized_grid(cfg, period: Period, norms: dict[str, dict]) -> pd.DataFrame:
    return apply_normalizers(period_grid(cfg, period), norms[period.symbol])


def training_frames(cfg, unit: Unit, norms: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.concat([normalized_grid(cfg, p, norms) for p in unit.train], ignore_index=True)
    val = pd.concat([normalized_grid(cfg, p, norms) for p in unit.validate], ignore_index=True)
    return train, val


# ----------------------------------------------------------- market context inputs

def sigma_daily_bps(frame: pd.DataFrame) -> float:
    """Daily vol from 5-minute mid returns (within each day), in bps."""
    t = frame[["ts", "mid"]].dropna().set_index("ts")["mid"]
    r = []
    for _, day in t.groupby(t.index.normalize()):
        five = day.resample("5min").last().dropna()
        r.append(np.log(five).diff().dropna() * 1e4)
    r = pd.concat(r)
    return float(r.std() * np.sqrt(78))


def symbol_sigma(cfg, unit: Unit, symbol: str) -> float:
    return sigma_daily_bps(norm_frame(cfg, unit, symbol))


def prior_dates(unit: Unit, period: Period) -> list[str]:
    """Dates of this symbol that the unit may learn from before `period`."""
    periods = unit.train + unit.validate + unit.norm.get(period.symbol, [])
    return sorted({p.date for p in periods if p.symbol == period.symbol and p.date < period.date})


def volume_curve(cfg, unit: Unit, period: Period) -> pd.Series:
    """Expected share of volume per 5-minute bucket, from days BEFORE `period`."""
    src = cfg["volume_curve"]["source"]
    if src == "flat":
        return flat_curve()
    if src == "proxy_csv":
        path = Path(cfg["volume_curve"]["path"].format(symbol=period.symbol))
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/fetch_volume_curve.py "
                                    f"--config {cfg.get('_path', '<config>')} first")
        return load_proxy_curve(path)
    if src == "past_days":
        prior = prior_dates(unit, period)
        if not prior:
            raise ValueError(f"no earlier days for a past-days volume curve on {period.date}")
        return curve_from_days([pd.Series(load_meta(cfg, period.symbol, d)["bucket_volume"])
                                for d in prior])
    raise ValueError(f"unknown volume curve source {src}")


def reference_daily_volume(cfg, unit: Unit, period: Period) -> float:
    """Stage 1: the same day's Nasdaq volume (a simulator setting no strategy sees).
    Stage 2: the average of earlier days."""
    if cfg["split"]["mode"] == "intraday":
        return float(load_meta(cfg, period.symbol, period.date)["total_volume"])
    prior = prior_dates(unit, period)
    return float(np.mean([load_meta(cfg, period.symbol, d)["total_volume"] for d in prior]))


def prediction_lookup(grid: pd.DataFrame, preds: np.ndarray):
    return as_ns(grid["ts"]), np.asarray(preds, dtype=float)
