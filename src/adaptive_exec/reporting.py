"""Tables and the pre-registered success test, computed from backtest output only.

Used by scripts/report.py and notebooks/03_results.ipynb.
"""
import json
import re
from pathlib import Path

import pandas as pd

from .experiment import FAMILIES
from .metrics import block_bootstrap_ci

ADAPTIVE = FAMILIES[3:]
ML = FAMILIES[-2:]
KEYS = ["unit", "order_id", "impact", "fill_mode"]
COST_PARTS = ["spread_bps", "drift_bps", "impact_bps", "fees_bps", "opportunity_bps"]
MAIN_KINDS = ("per_stock", "walk_forward")


def load_test_orders(cfg) -> pd.DataFrame:
    df = pd.read_csv(Path(cfg["results_dir"]) / "test_orders.csv")
    start = pd.to_datetime(df["start"])
    block = cfg["bootstrap"]["block"]
    df["block"] = start.dt.date.astype(str) if block == "day" else start.dt.floor(block).astype(str)
    df["side_label"] = df["side"].map({1: "buy", -1: "sell"})
    return paired(df)


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """Each strategy's IS minus VWAP's IS on the same parent order and settings."""
    base = df[df["strategy"] == "VWAP"].set_index(KEYS)["is_bps"]
    d = df.set_index(KEYS)
    d["delta_vs_vwap"] = d["is_bps"] - base.reindex(d.index).to_numpy()
    return d.reset_index()


def primary(d: pd.DataFrame, cfg) -> pd.DataFrame:
    t = cfg["tuning"]
    return d[(d["impact"] == t["impact"]) & (d["fill_mode"] == t["fill_mode"])]


def ci(d: pd.DataFrame, cfg) -> tuple[float, float, float]:
    b = cfg["bootstrap"]
    return block_bootstrap_ci(d["delta_vs_vwap"], d["block"], b["n_boot"], seed=b["seed"])


def fmt_ci(m, lo, hi) -> str:
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------- test tables

def main_table(d: pd.DataFrame, cfg) -> str:
    rows = []
    for s in FAMILIES:
        x = d[d["strategy"] == s]
        if x.empty:
            continue
        delta = "baseline" if s == "VWAP" else fmt_ci(*ci(x, cfg))
        rows.append([s, f"{x['is_bps'].mean():.2f}", f"{x['is_bps'].median():.2f}",
                     f"{x['is_bps'].std():.2f}", f"{x['is_bps'].quantile(0.95):.2f}", delta,
                     f"{x['passive_share'].mean():.1%}"])
    return md_table(["Strategy", "Mean IS (bps)", "Median", "Std. dev.", "95th pct",
                     "Δ vs VWAP (95% CI)", "Share filled passively"], rows)


def decomposition_table(d: pd.DataFrame) -> str:
    g = d.groupby("strategy")[COST_PARTS + ["is_bps"]].mean()
    rows = [[s] + [f"{v:.2f}" for v in g.loc[s]] for s in FAMILIES if s in g.index]
    return md_table(["Strategy", "Spread", "Drift", "Impact", "Fees", "Opportunity", "Total IS"], rows)


def robustness_table(d: pd.DataFrame, by: str, cfg, label: str | None = None) -> str:
    rows = []
    for s in FAMILIES[2:]:
        for v, x in d[d["strategy"] == s].groupby(by):
            rows.append([s, v, len(x), fmt_ci(*ci(x, cfg))])
    return md_table(["Strategy", label or by, "n orders", "Δ vs VWAP (95% CI)"], rows)


def success(d_all: pd.DataFrame, cfg) -> tuple[str, dict]:
    """Apply the criterion fixed in the README, exactly."""
    t = cfg["tuning"]
    parts, verdicts = [], {}
    for s in ML:
        x = d_all[d_all["strategy"] == s]
        prim = x[(x["impact"] == t["impact"]) & (x["fill_mode"] == t["fill_mode"])]
        if prim.empty:
            continue
        m, lo, hi = ci(prim, cfg)
        checks = {f"95% CI of Δ vs VWAP excludes zero, below it ({t['impact']} impact, "
                  f"{t['fill_mode']} fills)": hi < 0}
        for side, name in ((1, "buys"), (-1, "sells")):
            checks[f"direction holds for {name}"] = bool(prim.loc[prim["side"] == side, "delta_vs_vwap"].mean() < 0)
        for imp in cfg["impact"]:
            if imp != t["impact"]:
                y = x[(x["impact"] == imp) & (x["fill_mode"] == t["fill_mode"])]
                checks[f"direction holds at {imp} impact"] = bool(y["delta_vs_vwap"].mean() < 0)
        ok = all(checks.values())
        verdicts[s] = {"better_than_vwap": ok, "delta": m, "ci": (lo, hi)}
        verdict = "better than VWAP" if ok else "not shown to be better than VWAP"
        parts.append(f"**{s}: {verdict}** (Δ {fmt_ci(m, lo, hi)} bps)\n\n" +
                     "\n".join(f"- [{'x' if v else ' '}] {c}" for c, v in checks.items()))
    return "\n\n".join(parts), verdicts


def loso_table(d_all: pd.DataFrame, cfg) -> str:
    """Model trained and tuned on the other stocks vs on the same stock, same test orders."""
    prim = primary(d_all, cfg)
    loso, own = prim[prim["kind"] == "loso"], prim[prim["kind"] == "per_stock"]
    rows = []
    for s in ADAPTIVE:
        for sym in sorted(loso["symbol"].unique()):
            a = loso[(loso["strategy"] == s) & (loso["symbol"] == sym)]
            b = own[(own["strategy"] == s) & (own["symbol"] == sym)]
            rows.append([s, sym, len(a), fmt_ci(*ci(a, cfg)), fmt_ci(*ci(b, cfg)) if len(b) else "—"])
        a, b = loso[loso["strategy"] == s], own[own["strategy"] == s]
        rows.append([s, "**all**", len(a), fmt_ci(*ci(a, cfg)), fmt_ci(*ci(b, cfg)) if len(b) else "—"])
    return md_table(["Strategy", "Held-out stock", "n orders", "Δ vs VWAP, trained on other stocks",
                     "Δ vs VWAP, trained on the same stock"], rows)


# --------------------------------------------------------------- validation tables

def validation_tables(cfg) -> str:
    res = Path(cfg["results_dir"])
    tp, cp = res / "validation_tuning.csv", res / "chosen_params.json"
    if not (tp.exists() and cp.exists()):
        return "_No validation output found._"
    tuning, chosen = pd.read_csv(tp), json.loads(cp.read_text())
    main = tuning[tuning["kind"].isin(MAIN_KINDS)]
    picked = []
    for unit, fams in chosen.items():
        if unit not in set(main["unit"]):
            continue
        picked.append([unit, fams["Almgren-Chriss"].get("kappa_T"),
                       json.dumps(fams["Heuristic adaptive"]), json.dumps(fams["ML adaptive (tactic only)"]),
                       json.dumps(fams["ML adaptive (tactic + speed)"])])
    t1 = md_table(["Unit", "Almgren–Chriss κT", "Heuristic", "ML (tactic only)", "ML (tactic + speed)"], picked)

    # Chosen setting of every family, validation cost decomposition pooled over main units
    rows = []
    for fam in FAMILIES:
        sel = []
        for unit, fams in chosen.items():
            p = json.dumps(fams[fam], sort_keys=True)
            sel.append(main[(main["unit"] == unit) & (main["family"] == fam) & (main["params"] == p)])
        x = pd.concat(sel)
        if x.empty:
            continue
        w = x["n_orders"] / x["n_orders"].sum()
        rows.append([fam, int(x["n_orders"].sum())] +
                    [f"{(x[f'mean_{c}'] * w).sum():.2f}" for c in ["is_bps"] + COST_PARTS] +
                    [f"{(x['mean_passive_share'] * w).sum():.1%}"])
    t2 = md_table(["Strategy (chosen setting)", "n orders", "Mean IS", "Spread", "Drift", "Impact",
                   "Fees", "Opportunity", "Passive share"], rows)
    return ("### Settings chosen on validation\n\n" + t1 +
            "\n\n### Validation cost of the chosen settings (mean bps; these data chose them, "
            "so they are optimistic)\n\n" + t2)


def prediction_table(cfg) -> str:
    pm = Path(cfg["results_dir"]) / "prediction_metrics.csv"
    if not pm.exists():
        return "_No prediction metrics found._"
    p = pd.read_csv(pm)
    p = p[(p["split"] == "validate") & p["kind"].isin(MAIN_KINDS)]
    g = p.groupby(["model", "horizon_s"])[["r2", "corr", "hit_rate_nonzero", "share_zero_moves"]].mean()
    rows = [[m, f"{h} s", f"{r['r2']:.4f}", f"{r['corr']:+.4f}", f"{r['hit_rate_nonzero']:.3f}",
             f"{r['share_zero_moves']:.1%}"] for (m, h), r in g.iterrows()]
    return md_table(["Model", "Horizon", "R²", "Corr", "Sign hit rate (non-zero moves)",
                     "Share of zero moves"], rows)


def scoring_log(cfg) -> pd.DataFrame:
    p = Path(cfg["results_dir"]) / "scoring_log.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def assumptions(cfg) -> str:
    """Every proxy and simulator assumption, stated from the config."""
    vc = cfg["volume_curve"]
    curve = {"proxy_csv": f"PROXY: average share of day per 5-minute bucket from recent Yahoo Finance "
                          f"consolidated 5-minute bars (`{vc.get('path', '')}`), not the traded period's venue volume",
             "past_days": "average share of day per 5-minute bucket over the earlier days of the same stock",
             "flat": "flat (no intraday shape), so TWAP and VWAP coincide"}[vc["source"]]
    ref = ("the same day's total Nasdaq volume (a simulator setting that no strategy sees)"
           if cfg["split"]["mode"] == "intraday" else "the average daily volume of earlier days")
    imp = "; ".join(f"{k}: Y = {v['Y']}, half-life {v['half_life_s'] / 60:g} min" for k, v in cfg["impact"].items())
    f = cfg["fees"]
    return "\n".join([
        f"- **Data:** {cfg['source']} order book events, {', '.join(cfg['symbols'])}; Nasdaq only.",
        f"- **Volume curve for VWAP:** {curve}.",
        f"- **Parent order sizes:** {', '.join(f'{x:.0%}' for x in cfg['orders']['targets'])} of expected "
        f"volume over the {cfg['orders']['horizon_min']}-minute horizon; expected volume uses {ref}.",
        f"- **Fees:** taker ${f['taker_per_share']:.4f} per share, maker rebate ${f['maker_rebate_per_share']:.4f} "
        f"per share (config values; check them against the venue's schedule for the period).",
        f"- **Impact overlay** (a model, not a measurement): {imp}. Calibrated per parent order so the VWAP "
        f"schedule peaks at Y × σ_daily × √(Q / V_daily); σ_daily from each stock's training rows.",
        "- **Resting fills:** conservative = back of the queue, cancellations never move us forward; "
        "optimistic = every cancellation at our price is assumed to be ahead of us.",
        "- **Not modeled:** the market's reaction to our orders beyond the overlay, other venues, "
        "queue position from order IDs, auctions.",
    ])


# ------------------------------------------------------------------------ outputs

def build_summary(cfg) -> str:
    d_all = load_test_orders(cfg)
    d_main = d_all[d_all["kind"].isin(MAIN_KINDS)]
    prim = primary(d_main, cfg)
    t = cfg["tuning"]
    days = sorted(d_main["date"].astype(str).unique())
    n_orders = int((prim["strategy"] == "VWAP").sum())
    log = scoring_log(cfg)
    criterion, _ = success(d_main, cfg)
    parts = [
        f"# Results: {cfg['stage']}\n",
        *([f"> **{cfg['report_note']}**\n"] if cfg.get("report_note") else []),
        f"Generated by `scripts/report.py` from `test_orders.csv`. Out-of-sample test "
        f"{'days' if len(days) > 1 else 'day'}: {', '.join(days)}. Symbols: "
        f"{', '.join(sorted(d_main['symbol'].unique()))}. {n_orders} paired parent orders per strategy "
        f"and setting. IS = implementation shortfall vs the arrival mid, after fees; positive = cost. "
        f"Δ vs VWAP < 0 means cheaper than VWAP. 95% CIs: block bootstrap, blocks = "
        f"{cfg['bootstrap']['block']}.\n",
        f"Test periods scored {len(log)} time(s)." + (" Re-scoring reasons: " + "; ".join(
            f"{r.scored_at_utc}: {r.reason}" for r in log.itertuples() if r.rescore) if len(log) and
            log["rescore"].any() else "") + "\n",
        f"## Main table: {t['impact']} impact, {t['fill_mode']} fills\n", main_table(prim, cfg),
        "\n## Success criterion (fixed before testing)\n", criterion,
        "\n## Cost decomposition (mean bps)\n", decomposition_table(prim),
        "\n## Robustness\n",
        "### By impact setting (conservative fills)\n",
        robustness_table(d_main[d_main["fill_mode"] == "conservative"], "impact", cfg),
        f"\n### By fill assumption ({t['impact']} impact)\n",
        robustness_table(d_main[d_main["impact"] == t["impact"]], "fill_mode", cfg, "fills"),
        "\n### By side\n", robustness_table(prim, "side_label", cfg, "side"),
        "\n### By order size (share of expected volume)\n",
        robustness_table(prim, "size_target", cfg, "size"),
        "\n### By stock\n", robustness_table(prim, "symbol", cfg, "stock"),
    ]
    if (d_all["kind"] == "loso").any():
        parts += ["\n## Leave-one-stock-out\n",
                  "Models, thresholds and speed steps chosen on the other stocks, then applied to the "
                  f"held-out stock's test window ({t['impact']} impact, {t['fill_mode']} fills).\n",
                  loso_table(d_all, cfg)]
    parts += [
        "\n## Assumptions\n", assumptions(cfg),
        "\n## Validation\n", validation_tables(cfg),
        "\n## Prediction metrics (validation, mean over units; secondary)\n", prediction_table(cfg),
        "\n## Checks\n",
        f"- Mean realized participation (VWAP): {prim.loc[prim['strategy'] == 'VWAP', 'participation'].mean():.1%}",
        f"- Parent orders with shares the displayed book could not absorb at the horizon: "
        f"{(prim['unfilled'] > 0).mean():.1%}",
        "- Mean catch-up crossings per order: " + ", ".join(
            f"{s} {prim.loc[prim['strategy'] == s, 'n_catchups'].mean():.1f}" for s in FAMILIES
            if (prim["strategy"] == s).any()),
    ]
    return "\n".join(parts) + "\n"


def readme_block(cfg) -> str:
    """Compact results for the README, between the stage's markers."""
    d_all = load_test_orders(cfg)
    d_main = d_all[d_all["kind"].isin(MAIN_KINDS)]
    prim = primary(d_main, cfg)
    _, verdicts = success(d_main, cfg)
    t = cfg["tuning"]
    days = sorted(d_main["date"].astype(str).unique())
    n = int((prim["strategy"] == "VWAP").sum())
    lines = [*([f"> {cfg['report_note']}\n"] if cfg.get("report_note") else []),
             f"Out-of-sample: {n} paired parent orders on {', '.join(sorted(d_main['symbol'].unique()))}, "
             f"test {'days' if len(days) > 1 else 'day'} {', '.join(days)}; {t['impact']} impact, "
             f"{t['fill_mode']} fills. Generated by `scripts/report.py`; full tables in "
             f"`results/{Path(cfg['results_dir']).name}/summary.md`.\n", main_table(prim, cfg), ""]
    for s, v in verdicts.items():
        lines.append(f"- **{s}:** {'better than VWAP' if v['better_than_vwap'] else 'not shown to be better than VWAP'} "
                     f"under the pre-registered criterion (Δ {fmt_ci(v['delta'], *v['ci'])} bps).")
    return "\n".join(lines)


def inject_readme(readme: Path, stage: str, block: str) -> bool:
    text = readme.read_text()
    pat = re.compile(rf"(<!-- BEGIN RESULTS: {re.escape(stage)} -->\n).*?(\n<!-- END RESULTS: {re.escape(stage)} -->)",
                     re.S)
    if not pat.search(text):
        return False
    readme.write_text(pat.sub(lambda m: m.group(1) + block + m.group(2), text))
    return True
