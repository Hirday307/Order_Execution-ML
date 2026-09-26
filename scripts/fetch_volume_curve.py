"""Stage 1 proxy intraday volume curve from Yahoo 5-minute bars (last 60 days only).

    python scripts/fetch_volume_curve.py --config configs/stage1_lobster.yaml

This is recent consolidated volume, not 2012 Nasdaq volume. Only its U-shape
matters. Label it as a proxy in the write-up.
"""
import argparse
from pathlib import Path

import yfinance as yf

from adaptive_exec.pipeline import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for symbol in cfg["symbols"]:
        bars = yf.Ticker(symbol).history(period="60d", interval="5m")   # intraday bars: last 60 days only
        bars = bars[bars["Volume"] > 0]
        day_total = bars.groupby(bars.index.date)["Volume"].transform("sum")
        share = bars["Volume"] / day_total
        curve = share.groupby(bars.index.strftime("%H:%M")).mean()
        curve = (curve / curve.sum()).rename("share")
        path = Path(cfg["volume_curve"]["path"].format(symbol=symbol))
        path.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(path)
        print(f"{symbol}: {bars.index.normalize().nunique()} days -> {path}")


if __name__ == "__main__":
    main()
