"""Write synthetic days in LOBSTER's file format, for a dry run of the whole pipeline.

    python scripts/make_synthetic_lobster.py --out data/raw/synthetic --symbols SYNA SYNB SYNC

The data has a planted signal. Results on it validate plumbing, never a strategy.
"""
import argparse

from adaptive_exec.synthetic import write_lobster_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--symbols", nargs="+", default=["SYNA", "SYNB", "SYNC"])
    ap.add_argument("--date", default="2012-06-21")
    ap.add_argument("--rate", type=float, default=6.0, help="events per second (average)")
    args = ap.parse_args()
    for i, sym in enumerate(args.symbols):
        write_lobster_day(args.out, sym, args.date, seed=i, rate_per_s=args.rate,
                          start_price=30.0 + 20 * i)
        print(f"wrote {sym} {args.date} to {args.out}")


if __name__ == "__main__":
    main()
