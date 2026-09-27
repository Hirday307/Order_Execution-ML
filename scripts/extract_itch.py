"""Pass 1 for Nasdaq ITCH 5.0 sample files: keep the chosen symbols' order messages.

    python scripts/extract_itch.py --config configs/stage1_itch.yaml

Streams each gzipped file named in `itch_files` (about 5.5 GB compressed, 12 GB
raw) once, and writes `{raw_dir}/{date}/{symbol}.npz`. `build_dataset.py` then
rebuilds each symbol's book from those messages.
"""
import argparse
import time
from pathlib import Path

from adaptive_exec.itch import extract_messages, itch_date
from adaptive_exec.pipeline import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for name in cfg["itch_files"]:
        gz = Path(cfg["raw_dir"]) / name
        out = Path(cfg["raw_dir"]) / itch_date(gz)
        t0 = time.time()
        counts = extract_messages(gz, cfg["symbols"], out)
        print(f"[itch] {gz.name}: " + ", ".join(f"{s} {n:,}" for s, n in counts.items()) +
              f" messages kept in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
