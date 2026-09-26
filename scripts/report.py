"""results/<stage>/test_orders.csv -> results/<stage>/summary.md (and optionally the README).

Every number in the summary is computed here from the backtest output.

    python scripts/report.py --config configs/stage1_lobster.yaml
    python scripts/report.py --config configs/stage1_lobster.yaml --readme README.md
"""
import argparse
from pathlib import Path

from adaptive_exec.pipeline import load_config
from adaptive_exec.reporting import build_summary, inject_readme, readme_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--readme", help="also refresh this README's results block for the stage")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["results_dir"]) / "summary.md"
    summary = build_summary(cfg)
    out.write_text(summary)
    print(summary)
    print(f"[report] wrote {out}")
    if args.readme:
        block = cfg.get("readme_block", cfg["stage"])
        if inject_readme(Path(args.readme), block, readme_block(cfg)):
            print(f"[report] refreshed the '{block}' results block in {args.readme}")
        else:
            print(f"[report] no '<!-- BEGIN RESULTS: {block} -->' markers in {args.readme}; not changed")


if __name__ == "__main__":
    main()
