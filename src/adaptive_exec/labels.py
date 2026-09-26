"""Target: mid-price move over the next `horizon_s` seconds, in bps.

This is the only place that is allowed to look forward in time.
"""
import numpy as np
import pandas as pd

from .features import NS, as_ns


def forward_move_bps(events: pd.DataFrame, grid: pd.DatetimeIndex, horizon_s: int = 30) -> np.ndarray:
    """(mid at g + h) / (mid at g) - 1, in bps, using the last valid book at or before
    each time. NaN when g + h is past the last event."""
    valid = events["valid"].to_numpy()
    v_ns = as_ns(events["ts"])[valid]
    mid = events["mid"].to_numpy()[valid]
    g = as_ns(grid)
    ahead = g + horizon_s * NS
    k_now = np.searchsorted(v_ns, g, side="right") - 1
    k_ahead = np.searchsorted(v_ns, ahead, side="right") - 1
    y = np.full(len(g), np.nan)
    ok = (k_now >= 0) & (ahead <= v_ns[-1])
    y[ok] = (mid[k_ahead[ok]] / mid[k_now[ok]] - 1) * 1e4
    return y
