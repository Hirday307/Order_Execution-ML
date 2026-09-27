"""18 point-in-time features on a 1-second grid.

A grid row at time g uses only events with ts <= g: the last valid book state at
or before g, and flow summed over (g - w, g]. Lookups use `np.searchsorted(...,
side="right")`, which is a backward as-of join. `tests/test_no_lookahead.py`
checks that deleting future events never changes a past feature.

Features 7-9 (OFI) and 13 (relative volume) are scaled by normalizers that must
be fitted on training data only (`fit_normalizers`).
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "spread_ticks", "spread_bps",
    "imb_l1", "imb_l5", "micro_gap_bps", "depth_l5_log",
    "ofi_10s", "ofi_30s", "ofi_60s",
    "trade_imb_30s", "trade_imb_60s",
    "trade_count_60s", "volume_60s_rel",
    "ret_10s", "ret_60s", "ret_300s",
    "rv_300s",
    "minutes_since_open",
]
NS = 1_000_000_000


def as_ns(ts) -> np.ndarray:
    return np.asarray(pd.to_datetime(ts).astype("datetime64[ns]")).view("int64")


def make_grid(events: pd.DataFrame, freq: str = "1s") -> pd.DatetimeIndex:
    """Whole-second grid from the first to the last event of the frame."""
    ts = events["ts"]
    return pd.date_range(ts.iloc[0].ceil(freq), ts.iloc[-1].floor(freq), freq=freq, unit="ns")


def _asof_index(event_ns: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    """Index of the last event with ts <= query, or -1."""
    return np.searchsorted(event_ns, query_ns, side="right") - 1


def _take(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def _window_sum(event_ns: np.ndarray, values: np.ndarray, grid_ns: np.ndarray, w_ns: int) -> np.ndarray:
    """Sum of values for events with ts in (g - w, g]."""
    csum = np.concatenate([[0.0], np.cumsum(values, dtype=float)])
    hi = np.searchsorted(event_ns, grid_ns, side="right")
    lo = np.searchsorted(event_ns, grid_ns - w_ns, side="right")
    return csum[hi] - csum[lo]


def _ratio(num, den):
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def build_raw_features(events: pd.DataFrame, grid: pd.DatetimeIndex | None = None,
                       tick_size: float = 0.01) -> pd.DataFrame:
    """Features before normalization, plus `mid` and `l1_depth` (used for normalizers)."""
    grid = make_grid(events) if grid is None else grid
    g = as_ns(grid)
    ev_ns = as_ns(events["ts"])

    # ---- book state: last VALID (two-sided, uncrossed) book at or before g
    valid = events["valid"].to_numpy()
    need = ["bid_px_1", "ask_px_1", "bid_sz_1", "ask_sz_1"] + \
           [f"{s}_sz_{i}" for s in ("bid", "ask") for i in range(1, 6)]
    v = events.loc[valid, list(dict.fromkeys(need))]
    v_ns = ev_ns[valid]
    k = _asof_index(v_ns, g)
    bid, ask = v["bid_px_1"].to_numpy(), v["ask_px_1"].to_numpy()
    bsz, asz = v["bid_sz_1"].to_numpy(), v["ask_sz_1"].to_numpy()
    mid = (bid + ask) / 2
    b5 = v[[f"bid_sz_{i}" for i in range(1, 6)]].fillna(0).sum(axis=1).to_numpy()
    a5 = v[[f"ask_sz_{i}" for i in range(1, 6)]].fillna(0).sum(axis=1).to_numpy()
    wmid = (ask * bsz + bid * asz) / (bsz + asz)

    f = pd.DataFrame(index=pd.RangeIndex(len(g)))
    f["ts"] = grid
    m = _take(mid, k)
    f["mid"] = m
    f["spread_ticks"] = _take((ask - bid) / tick_size, k)
    f["spread_bps"] = _take((ask - bid) / mid * 1e4, k)
    f["imb_l1"] = _take((bsz - asz) / (bsz + asz), k)
    f["imb_l5"] = _take(_ratio(b5 - a5, b5 + a5), k)
    f["micro_gap_bps"] = _take((wmid - mid) / mid * 1e4, k)
    f["depth_l5_log"] = _take(np.log1p(b5 + a5), k)
    f["l1_depth"] = _take((bsz + asz) / 2, k)

    # ---- order flow imbalance (Cont, Kukanov & Stoikov 2014) between valid books
    bp_prev, bq_prev = np.roll(bid, 1), np.roll(bsz, 1)
    ap_prev, aq_prev = np.roll(ask, 1), np.roll(asz, 1)
    e = ((bid >= bp_prev) * bsz - (bid <= bp_prev) * bq_prev
         - (ask <= ap_prev) * asz + (ask >= ap_prev) * aq_prev)
    if len(e):
        e[0] = 0.0
    e = np.nan_to_num(e)
    for w in (10, 30, 60):
        f[f"ofi_{w}s_raw"] = _window_sum(v_ns, e, g, w * NS)

    # ---- trades
    tr = events["is_trade"].to_numpy()
    t_ns = ev_ns[tr]
    size = events["size"].to_numpy(dtype=float)[tr]
    agg = events["aggressor"].to_numpy()[tr]
    for w in (30, 60):
        buy = _window_sum(t_ns, size * (agg > 0), g, w * NS)
        sell = _window_sum(t_ns, size * (agg < 0), g, w * NS)
        f[f"trade_imb_{w}s"] = _ratio(buy - sell, buy + sell)
    f["trade_count_60s"] = _window_sum(t_ns, np.ones_like(size), g, 60 * NS)
    f["volume_60s_raw"] = _window_sum(t_ns, size, g, 60 * NS)

    # ---- returns and volatility
    for w in (10, 60, 300):
        past = _take(mid, _asof_index(v_ns, g - w * NS))
        f[f"ret_{w}s"] = (m / past - 1) * 1e4
    r1 = pd.Series(np.log(m)).diff() * 1e4
    f["rv_300s"] = r1.rolling(300, min_periods=300).std().to_numpy()

    open_ns = as_ns(grid.normalize() + pd.Timedelta(hours=9, minutes=30))
    f["minutes_since_open"] = (g - open_ns) / (60 * NS)
    return f


def fit_normalizers(raw_train: pd.DataFrame) -> dict:
    """Scales for OFI and relative volume, from TRAINING rows only."""
    return {
        "l1_depth": float(np.nanmean(raw_train["l1_depth"])),
        "volume_60s": float(np.nanmean(raw_train["volume_60s_raw"])),
    }


IDENTITY_NORMALIZERS = {"l1_depth": 1.0, "volume_60s": 1.0}


def apply_normalizers(raw: pd.DataFrame, norm: dict) -> pd.DataFrame:
    f = raw.copy()
    for w in (10, 30, 60):
        f[f"ofi_{w}s"] = raw[f"ofi_{w}s_raw"] / norm["l1_depth"]
    f["volume_60s_rel"] = raw["volume_60s_raw"] / norm["volume_60s"]
    return f


def build_features(events: pd.DataFrame, normalizers: dict | None = None,
                   grid: pd.DatetimeIndex | None = None, tick_size: float = 0.01) -> pd.DataFrame:
    """`ts` plus FEATURE_COLUMNS. Without normalizers, scales are left at 1."""
    raw = build_raw_features(events, grid, tick_size)
    f = apply_normalizers(raw, normalizers or IDENTITY_NORMALIZERS)
    return f[["ts"] + FEATURE_COLUMNS]
