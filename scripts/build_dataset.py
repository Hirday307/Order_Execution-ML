"""Raw LOBSTER / Databento files -> processed events, 1 s feature grid with labels, and metadata.

    python scripts/build_dataset.py --config configs/stage1_lobster.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_exec.features import build_raw_features
from adaptive_exec.labels import forward_move_bps
from adaptive_exec.loaders import FILL_REMOVAL, find_lobster_files, load_databento, load_lobster
from adaptive_exec.model import label_column, save_json
from adaptive_exec.pipeline import day_dir, load_config
from adaptive_exec.schedules import bucket_volumes
from adaptive_exec.splits import at


def raw_days(cfg, symbol):
    """Yields (date, loader) for every raw day of a symbol."""
    if cfg["source"] == "lobster":
        for date, msg, book in find_lobster_files(cfg["raw_dir"], symbol, cfg["levels"]):
            yield date, lambda m=msg, b=book, d=date: load_lobster(m, b, d, cfg["levels"])
    elif cfg["source"] == "databento":
        lo, hi = cfg["download"]["start"], cfg["download"]["end"]    # this stage's days only
        for path in sorted((Path(cfg["raw_dir"]) / symbol).glob(f"{symbol}_*.dbn.zst")):
            date = path.name.split("_")[1].split(".")[0]
            if lo <= date <= hi:
                yield date, lambda p=path: load_databento(p, cfg["levels"])
    else:
        raise ValueError(f"unknown source {cfg['source']}")


def sanity_checks(events: pd.DataFrame) -> dict:
    ts = events["ts"]
    secs = (ts - ts.dt.normalize()).dt.total_seconds()
    execs = events.index[events["type"] == 4]
    prev = execs[execs > 0] - 1
    e = events.loc[prev + 1]
    p = events.loc[prev]
    best = np.where(e["aggressor"].to_numpy() > 0, p["ask_px_1"].to_numpy(), p["bid_px_1"].to_numpy())
    tr = events[events["is_trade"]]
    vol = float(tr["size"].sum())
    return {
        "n_events": int(len(events)),
        "first_second": float(secs.min()), "last_second": float(secs.max()),
        "time_monotonic": bool(ts.is_monotonic_increasing),
        "mid_min": float(events["mid"].min()), "mid_max": float(events["mid"].max()),
        "share_invalid_book": float(1 - events["valid"].mean()),
        "n_trades": int(events["is_trade"].sum()),
        "share_visible_exec_at_prior_best": float(np.mean(np.isclose(e["price"].to_numpy(), best))),
        "share_trade_volume_unknown_aggressor": float(tr.loc[tr["aggressor"] == 0, "size"].sum() / vol)
        if vol else float("nan"),
        # Databento only: cancels re-labelled as the book update after a trade
        "fill_removals_per_trade": float((events["type"] == FILL_REMOVAL).sum() / max(len(tr), 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    for symbol in cfg["symbols"]:
        n_days = 0
        for date, load in raw_days(cfg, symbol):
            events = load()
            if events["valid"].sum() < 100 or not events["is_trade"].any():
                print(f"[build] {symbol} {date}: skipped, no usable book (holiday or empty file?)")
                continue
            checks = sanity_checks(events)
            grid = pd.date_range(at(date, "09:30"), at(date, "16:00"), freq="1s", unit="ns",
                                 inclusive="left")
            raw = build_raw_features(events, grid, cfg["tick_size"])
            for h in cfg["label"]["horizons"]:
                raw[label_column(h)] = forward_move_bps(events, grid, h)

            out = day_dir(cfg, symbol, date)
            out.mkdir(parents=True, exist_ok=True)
            events.to_parquet(out / "events.parquet", index=False)
            raw.to_parquet(out / "grid.parquet", index=False)
            buckets = bucket_volumes(events)
            save_json({"symbol": symbol, "date": date, "total_volume": float(buckets.sum()),
                       "bucket_volume": buckets.to_dict(), "sanity": checks}, out / "meta.json")
            n_days += 1
            print(f"[build] {symbol} {date}: {checks['n_events']:,} events, {checks['n_trades']:,} trades, "
                  f"mid {checks['mid_min']:.2f}-{checks['mid_max']:.2f}, "
                  f"invalid book {checks['share_invalid_book']:.2%}, "
                  f"exec at prior best {checks['share_visible_exec_at_prior_best']:.2%}, "
                  f"unknown-aggressor volume {checks['share_trade_volume_unknown_aggressor']:.2%}, "
                  f"monotonic={checks['time_monotonic']}")
        if n_days == 0:
            print(f"[build] WARNING: no raw files for {symbol} in {cfg['raw_dir']}")


if __name__ == "__main__":
    main()
