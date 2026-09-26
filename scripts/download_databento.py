"""Stage 2 download: Databento XNAS.ITCH mbp-10, one file per symbol and day.

    python scripts/download_databento.py --config configs/stage2_databento.yaml            # cost only
    python scripts/download_databento.py --config configs/stage2_databento.yaml --confirm  # download

Reads the API key from the DATABENTO_API_KEY environment variable. Never put it
in code or Git. The cost estimate always runs first.
"""
import argparse
from pathlib import Path

import pandas as pd

from adaptive_exec.pipeline import load_config


def session_utc(date: pd.Timestamp):
    ny = "America/New_York"
    start = pd.Timestamp(f"{date.date()} 09:30", tz=ny).tz_convert("UTC")
    end = pd.Timestamp(f"{date.date()} 16:00", tz=ny).tz_convert("UTC")
    return start.isoformat(), end.isoformat()


def main():
    import databento as db

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--confirm", action="store_true", help="actually download after the estimate")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["download"]
    client = db.Historical()
    days = pd.bdate_range(d["start"], d["end"])     # exchange holidays simply return no data

    jobs, total = [], 0.0
    for symbol in cfg["symbols"]:
        for day in days:
            start, end = session_utc(day)
            params = dict(dataset=d["dataset"], symbols=[symbol], schema=d["schema"], start=start, end=end)
            cost = client.metadata.get_cost(**params)
            total += cost
            jobs.append((symbol, day, params))
            print(f"{symbol} {day.date()}: ${cost:.2f}")
    print(f"Estimated total: ${total:.2f} for {len(jobs)} symbol-days")
    if not args.confirm:
        print("Re-run with --confirm to download.")
        return

    for symbol, day, params in jobs:
        out = Path(cfg["raw_dir"]) / symbol / f"{symbol}_{day.date()}.dbn.zst"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        client.timeseries.get_range(**params, path=out)
        print(f"downloaded {out}")


if __name__ == "__main__":
    main()
