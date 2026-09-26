"""Time splits with gaps (Stage 1) and day-level walk-forward (Stage 2)."""
from dataclasses import dataclass

import pandas as pd


def at(date, hhmm: str) -> pd.Timestamp:
    h, m = map(int, hhmm.split(":"))
    return pd.Timestamp(date).normalize() + pd.Timedelta(hours=h, minutes=m)


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp

    def mask(self, ts: pd.Series) -> pd.Series:
        return (ts >= self.start) & (ts < self.end)


def intraday_windows(date, split: dict, label_horizon_s: int) -> dict[str, Window]:
    """{'train', 'validate', 'test'} windows on one day. Gaps between consecutive
    windows must exceed the label horizon so no label crosses a boundary."""
    w = {name: Window(at(date, split[name][0]), at(date, split[name][1]))
         for name in ("train", "validate", "test")}
    for a, b in (("train", "validate"), ("validate", "test")):
        gap = (w[b].start - w[a].end).total_seconds()
        if gap <= label_horizon_s:
            raise ValueError(f"gap between {a} and {b} ({gap:.0f}s) must exceed the "
                             f"label horizon ({label_horizon_s}s)")
    return w


@dataclass(frozen=True)
class WalkForwardFold:
    train_days: tuple
    val_day: str
    test_day: str


def walk_forward(days: list[str], train_days: int = 10) -> list[WalkForwardFold]:
    """For each test day d: train on the `train_days` days before d-1, tune on d-1, test on d."""
    days = sorted(days)
    return [WalkForwardFold(tuple(days[i - 1 - train_days:i - 1]), days[i - 1], days[i])
            for i in range(train_days + 1, len(days))]
