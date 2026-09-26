"""Parent-order schedules (shares per slice) and intraday volume curves.

A volume curve is a Series indexed by 5-minute bucket start ("09:30" ... "15:55")
holding the expected share of the day's volume. It must come from days other
than the one being traded.
"""
import numpy as np
import pandas as pd

BUCKETS = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in range(0, 60, 5)
           if (h, m) >= (9, 30)]


def to_whole_shares(slices, total_qty):
    whole = np.floor(slices).astype(int)
    whole[-1] += total_qty - whole.sum()           # put the rounding remainder in the last slice
    return whole


def twap(total_qty, n_slices):
    return to_whole_shares(np.full(n_slices, total_qty / n_slices), total_qty)


def vwap(total_qty, curve):
    """curve: expected share of volume in each slice, from PAST days only."""
    w = np.asarray(curve, dtype=float)
    return to_whole_shares(total_qty * w / w.sum(), total_qty)


def almgren_chriss(total_qty, n_slices, kappa_T):
    """Front-loaded: shares remaining = X * sinh(kappa_T * (1 - t)) / sinh(kappa_T).
    kappa_T -> 0 gives TWAP; larger kappa_T trades more urgently."""
    if kappa_T < 1e-6:
        return twap(total_qty, n_slices)
    t = np.linspace(0.0, 1.0, n_slices + 1)
    remaining = total_qty * np.sinh(kappa_T * (1 - t)) / np.sinh(kappa_T)
    return to_whole_shares(-np.diff(remaining), total_qty)


# ------------------------------------------------------------------ volume curves

def flat_curve() -> pd.Series:
    return pd.Series(1.0 / len(BUCKETS), index=BUCKETS)


def load_proxy_curve(path) -> pd.Series:
    """CSV written by scripts/fetch_volume_curve.py: bucket "HH:MM", share."""
    s = pd.read_csv(path, index_col=0).iloc[:, 0]
    s.index = s.index.astype(str).str[:5]
    s = s.reindex(BUCKETS).fillna(0.0)
    return s / s.sum()


def bucket_volumes(events: pd.DataFrame) -> pd.Series:
    """Traded volume per 5-minute bucket for one day."""
    tr = events[events["is_trade"]]
    b = tr["ts"].dt.floor("5min").dt.strftime("%H:%M")
    return tr.groupby(b)["size"].sum().reindex(BUCKETS).fillna(0.0)


def curve_from_days(daily_bucket_volumes: list[pd.Series]) -> pd.Series:
    """Average share-of-day curve over earlier days."""
    shares = [v / v.sum() for v in daily_bucket_volumes if v.sum() > 0]
    s = pd.concat(shares, axis=1).mean(axis=1)
    return s / s.sum()


def slice_weights(curve: pd.Series, slice_starts: pd.DatetimeIndex, slice_s: int) -> np.ndarray:
    """Expected share of the day's volume in each slice (bucket share spread evenly)."""
    b = slice_starts.floor("5min").strftime("%H:%M")
    return curve.reindex(b).fillna(0.0).to_numpy() * slice_s / 300.0
