"""Build figures and ``reports/REPORT.md`` from saved results.

Every number in the report is computed here from files in ``reports/data``;
nothing is typed by hand, so re-running the pipeline regenerates the report.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from adlift import viz

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports" / "data"
FIG = ROOT / "reports" / "figures"


def _load(name):
    p = DATA / f"{name}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            if f and isinstance(v, (int, float, np.floating, np.integer)) and pd.notna(v):
                cells.append(f.format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pp(x):
    return f"{100 * x:+.2f}pp"


def section_incrementality_mc(out: list[str]):
    p = DATA / "incrementality_mc_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    viz.incrementality_mc(s, FIG / "incrementality_mc_conversion.png", "converted")
    viz.incrementality_mc(s, FIG / "incrementality_mc_awareness.png", "aware")
    reps = int(s["reps"].max())
    out.append("## 1. Incrementality estimators vs. known truth (Monte Carlo)\n")
    out.append(f"{reps} simulated campaigns per scenario, 20,000 users each. Target: ATT of exposure. "
               "Truth is exact (both potential outcomes are simulated).\n")
    out.append("![](figures/incrementality_mc_conversion.png)\n")
    for o, lab in [("converted", "Conversion (observed for all users)"), ("aware", "Brand awareness (survey respondents only)")]:
        t = s[s.outcome == o].copy()
        t["method"] = t["method"].map(viz.METHOD_LABEL)
        t = t[["scenario", "method", "truth", "bias", "relative_bias", "rmse", "coverage", "reject_rate"]]
        out.append(f"**{lab}**\n")
        out.append(md(t, {"truth": "{:.4f}", "bias": "{:+.4f}", "relative_bias": "{:+.0%}", "rmse": "{:.4f}",
                          "coverage": "{:.0%}", "reject_rate": "{:.0%}"}) + "\n")
    out.append("![](figures/incrementality_mc_awareness.png)\n")
    raw = DATA / "incrementality_mc_raw.csv"
    if raw.exists():
        r = pd.read_csv(raw)
        wc = r[r.method == "winners_curse_top_age_segment"]
        if len(wc):
            g = wc.groupby("scenario").agg(overstatement=("error", "mean"), truth=("truth", "mean"),
                                           top_is_truly_best=("true_rank", lambda x: float((x == 1).mean())),
                                           segment_ci_coverage=("segment_coverage", "mean")).reset_index()
            g["overstatement_pct"] = g["overstatement"] / g["truth"]
            out.append("**Winner's curse.** Picking the age segment with the largest estimated conversion lift:\n")
            out.append(md(g[["scenario", "overstatement", "overstatement_pct", "top_is_truly_best", "segment_ci_coverage"]],
                          {"overstatement": "{:+.4f}", "overstatement_pct": "{:+.0%}", "top_is_truly_best": "{:.0%}",
                           "segment_ci_coverage": "{:.0%}"}) + "\n")


def section_mmm_mc(out: list[str]):
    p = DATA / "mmm_mc_raw.csv"
    if not p.exists():
        return
    from adlift.benchmark.mmm_mc import summarize

    raw = pd.read_csv(p)
    s = summarize(raw)
    s.to_csv(DATA / "mmm_mc_summary.csv", index=False)
    viz.mmm_mc(s, FIG / "mmm_mc_search.png", "search")
    n = raw["seed"].nunique()
    out.append("## 2. Bayesian MMM vs. known truth (Monte Carlo)\n")
    out.append(f"{n} simulated advertisers x 156 weeks x 4 channels; 6 fitting conditions per seed. "
               "Truth: exact ROAS and marginal ROAS from the simulator.\n")
    out.append("![](figures/mmm_mc_search.png)\n")
    t = s.copy()
    out.append(md(t[["scenario", "calibration", "metric", "channel", "truth", "bias", "rmse", "coverage", "width"]],
                  {"truth": "{:.2f}", "bias": "{:+.2f}", "rmse": "{:.2f}", "coverage": "{:.0%}", "width": "{:.2f}"}) + "\n")
    g = raw.drop_duplicates(["seed", "scenario", "calibration"]).groupby(["scenario", "calibration"], sort=False)
    gain = g["share_of_achievable_gain"].agg(median="median", p_positive=lambda x: float((x > 0).mean())).reset_index()
    diag = g.agg(max_rhat=("max_rhat", "max"), median_divergences=("divergences", "median")).reset_index()
    out.append("**Budget reallocation scored with the true curves** (share of the truly achievable gain captured "
               "by the model's recommendation; negative = the recommendation loses money):\n")
    out.append(md(gain.merge(diag), {"median": "{:+.0%}", "p_positive": "{:.0%}", "max_rhat": "{:.3f}"}) + "\n")


def section_criteo(out: list[str]):
    r = _load("criteo")
    if r is None:
        return
    viz.criteo(r["observational"], FIG / "criteo_lalonde.png")
    out.append("## 3. Real data: Criteo randomized ad test (LaLonde-style)\n")
    q = r["quality"].set_index("check_name")["value"]
    out.append(f"Rows: {int(q['rows']):,}. Treatment share {q['treatment_share']:.4f} (design 0.85; SRM p = "
               f"{r['srm']['p_value']:.3g}). Exposed users in control: {int(q['exposed_in_control'])}. "
               f"Null cells: {int(q['null_cells'])}.\n")
    out.append("**Experimental benchmark (all users):**\n")
    bt = r["benchmark"][["outcome", "itt", "itt_se", "itt_relative", "exposure_rate_treated", "tot", "tot_se",
                         "tot_ci_low", "tot_ci_high", "tot_relative"]]
    out.append(md(bt, {"itt": "{:.5f}", "itt_se": "{:.5f}", "itt_relative": "{:+.1%}", "exposure_rate_treated": "{:.2%}",
                       "tot": "{:.4f}", "tot_se": "{:.4f}", "tot_ci_low": "{:.4f}", "tot_ci_high": "{:.4f}",
                       "tot_relative": "{:+.0%}"}) + "\n")
    out.append(f"**Observational re-analysis** (treatment arm only, {r['sample_n']:,} sampled users, "
               f"{r['sample_exposed']:,} exposed; control arm discarded):\n")
    out.append("![](figures/criteo_lalonde.png)\n")
    ob = r["observational"][["outcome", "method", "estimate", "se", "rct_tot", "ratio_to_rct", "z_vs_rct"]]
    out.append(md(ob, {"estimate": "{:.4f}", "se": "{:.4f}", "rct_tot": "{:.4f}", "ratio_to_rct": "{:.2f}x",
                       "z_vs_rct": "{:+.1f}"}) + "\n")
    ov = r["overlap"]
    out.append("Overlap (estimated propensity of exposure):\n")
    out.append(md(ov[["arm", "n", "p5", "p50", "p95", "p99", "share_below_0.05", "share_above_0.95"]],
                  {"p5": "{:.4f}", "p50": "{:.4f}", "p95": "{:.3f}", "p99": "{:.3f}", "share_below_0.05": "{:.1%}",
                   "share_above_0.95": "{:.1%}"}) + "\n")
    bal = r["balance"].head(6)
    out.append("Largest covariate imbalances (SMD, before / after ATT weighting):\n")
    out.append(md(bal[["covariate", "smd_unweighted", "smd_weighted"]], {"smd_unweighted": "{:+.2f}", "smd_weighted": "{:+.2f}"}) + "\n")
    for o, h in r.get("heterogeneity", {}).items():
        d, se = h.attrs.get("top_minus_bottom", (np.nan, np.nan))
        out.append(f"**Heterogeneity ({o}): ITT by quintile of baseline {o} propensity** "
                   f"(top minus bottom: {pp(d)} +/- {100 * 1.645 * se:.2f}pp):\n")
        out.append(md(h[["score_quintile", "n", "baseline_rate", "itt", "se", "relative", "qvalue_bh"]],
                      {"baseline_rate": "{:.4f}", "itt": "{:.5f}", "se": "{:.5f}", "relative": "{:+.0%}", "qvalue_bh": "{:.2g}"}) + "\n")


def section_ads_psa(out: list[str]):
    r = _load("ads_psa")
    if r is None:
        return
    viz.ads_psa(r["by_exposure_bin"], FIG / "ads_psa_activity_bias.png")
    o = r["overall"].iloc[0]
    out.append("## 4. Real data: ads vs. PSA experiment\n")
    out.append(f"{r['n_users']:,} users ({int(o['n_psa']):,} PSA). Lift {pp(o['lift'])} "
               f"(90% CI {pp(o['ci_low'])} to {pp(o['ci_high'])}), {o['relative']:+.0%} relative "
               f"({o['psa_rate']:.2%} -> {o['ad_rate']:.2%}).\n")
    out.append("![](figures/ads_psa_activity_bias.png)\n")
    ab = r["activity_bias"]
    out.append(f"**Activity bias.** Heavy- vs. light-exposure users (median split): ad arm "
               f"{pp(ab.set_index('arm').loc['ad', 'heavy_minus_light'])}, PSA arm "
               f"{pp(ab.set_index('arm').loc['psa', 'heavy_minus_light'])}. "
               f"{r['activity_bias_share']:.0%} of the naive gap appears where ads cannot work.\n")
    out.append(md(r["by_exposure_bin"][["ads_bin", "n", "n_psa", "psa_rate", "ad_rate", "lift", "se", "relative", "qvalue_bh"]],
                  {"psa_rate": "{:.4f}", "ad_rate": "{:.4f}", "lift": "{:+.4f}", "se": "{:.4f}", "relative": "{:+.0%}",
                   "qvalue_bh": "{:.2g}"}) + "\n")
    out.append("**Randomization checks and sensitivity.** Arms differ in day/hour mix, so lift is re-estimated with adjustment:\n")
    out.append(md(r["balance"], {"statistic": "{:.4g}", "pvalue": "{:.2g}"}) + "\n")
    out.append(md(r["day_mix"], {"psa_share": "{:.3f}", "ad_share": "{:.3f}", "difference_pp": "{:+.2f}"}) + "\n")
    out.append(md(r["sensitivity"], {"lift": "{:+.5f}", "se": "{:.5f}", "ci_low": "{:+.5f}", "ci_high": "{:+.5f}",
                                     "relative": "{:+.1%}", "pvalue": "{:.2g}"}) + "\n")


def section_pipeline(out: list[str]):
    r = _load("pipeline")
    if r is None:
        return
    out.append("## 5. Campaign measurement pipeline (SQL + diagnostics + brand lift)\n")
    out.append(f"One simulated 16-campaign flight, {r['n_users']:,} users, raw logs with injected defects, "
               "processed by the DuckDB SQL layer.\n")
    out.append("**Raw-data quality checks** (rows removed by staging):\n")
    out.append(md(r["raw_quality"], {"failing_rate": "{:.2%}"}) + "\n")
    out.append("**Analysis-table checks** (all must be 0):\n")
    out.append(md(r["analysis_quality"]) + "\n")
    out.append("**Missingness:**\n")
    m = r["missingness"]
    out.append(md(m, {c: "{:.3f}" for c in m.columns if c != "column"}) + "\n")
    sv = r["survey"]
    out.append(f"**Survey response:** {sv['response_rate_exposed']:.1%} of invited exposed users responded vs. "
               f"{sv['response_rate_unexposed']:.1%} unexposed (p = {sv['response_depends_on_exposure_p']:.2g}).\n")
    out.append("**Balance** (top covariates by |SMD|):\n")
    out.append(md(r["balance"].head(6)[["covariate", "smd_unweighted", "smd_weighted"]],
                  {"smd_unweighted": "{:+.2f}", "smd_weighted": "{:+.3f}"}) + "\n")
    eff = pd.DataFrame(r["effects"])
    out.append("**Effects vs. truth:**\n")
    out.append(md(eff[["outcome", "method", "estimate", "ci_low", "ci_high", "relative", "truth", "covered"]],
                  {"estimate": "{:+.4f}", "ci_low": "{:+.4f}", "ci_high": "{:+.4f}", "relative": "{:+.0%}", "truth": "{:+.4f}"}) + "\n")
    k = r["brand_lift_kpis"]
    out.append(f"**Brand-lift KPIs (awareness, AIPW-GBM):** absolute lift {k['absolute_lift_pp']:+.1f}pp; "
               f"lifted users {k['lifted_users']:,.0f} (90% CI {k['lifted_users_ci'][0]:,.0f} to {k['lifted_users_ci'][1]:,.0f}; "
               f"truth {k['true_lifted_users']:,.0f}); cost per lifted user ${k['cost_per_lifted_user']:.2f} "
               f"on ${k['spend_usd']:,.0f} spend.\n")
    for (o, by), t in r["segments"].items():
        if by in ("age_group", "campaign_id") and len(t):
            out.append(f"Segment effects - {o} by {by} (BH q-values; truth from simulator):\n")
            out.append(md(t[["segment", "n_exposed", "estimate", "ci_low", "ci_high", "qvalue_bh", "truth", "covered"]],
                          {"estimate": "{:+.4f}", "ci_low": "{:+.4f}", "ci_high": "{:+.4f}", "qvalue_bh": "{:.2g}",
                           "truth": "{:+.4f}"}) + "\n")
    for o, t in r["meta_regression"].items():
        out.append(f"Campaign meta-regression ({o}; 16 campaigns, descriptive):\n")
        out.append(md(t, {"coef": "{:+.4f}", "se": "{:.4f}", "z": "{:+.1f}"}) + "\n")
    fq = r["frequency"]["naive"].merge(r["frequency"]["true"], on="frequency", suffixes=("", "_true"))
    out.append("**Frequency:** naive (exposed at frequency f minus unexposed) vs. true effect at f:\n")
    out.append(md(fq[["frequency", "n", "naive_effect", "true_effect"]], {"naive_effect": "{:+.4f}", "true_effect": "{:+.4f}"}) + "\n")


def section_meridian(out: list[str]):
    r = _load("meridian")
    if r is None:
        return
    viz.response_curves(r["curves"], FIG / "meridian_response_curves.png")
    v = r["validation"]
    out.append("## 6. Bayesian MMM on Google Meridian's public data\n")
    out.append(f"156 weeks, 5 channels, aggregated from 40 geos in SQL. Diagnostics: max R-hat "
               f"{r['fit_diagnostics']['max_rhat']:.3f}, min ESS {r['fit_diagnostics']['min_ess']:.0f}, "
               f"divergences {r['fit_diagnostics']['divergences']}. Train MAPE {v['train_mape']:.1%}; "
               f"13-week holdout MAPE {v['holdout_mape']:.1%} vs. {v['holdout_mape_naive_last13_mean']:.1%} for a naive forecast. "
               f"Media share of revenue: {r['media_share_of_revenue']:.0%}.\n")
    s = r["summary"]
    out.append(md(s[["channel", "spend_share", "roas_mean", "roas_low", "roas_high", "mroas_mean", "mroas_low", "mroas_high",
                     "p_mroas_above_1"]],
                  {"spend_share": "{:.0%}", "roas_mean": "{:.2f}", "roas_low": "{:.2f}", "roas_high": "{:.2f}",
                   "mroas_mean": "{:.2f}", "mroas_low": "{:.2f}", "mroas_high": "{:.2f}", "p_mroas_above_1": "{:.0%}"}) + "\n")
    out.append("![](figures/meridian_response_curves.png)\n")
    g = r["reallocation_gain"]
    out.append(f"**Reallocation (same budget, each channel within +/-50%):** expected +${g['mean']:,.0f}/week "
               f"({g['pct_of_revenue']:+.2%} of revenue), 90% interval ${g['ci'][0]:,.0f} to ${g['ci'][1]:,.0f}, "
               f"P(gain > 0) = {g['p_positive']:.0%}.\n")
    out.append(md(r["allocation"], {"current_weekly": "${:,.0f}", "recommended_weekly": "${:,.0f}", "change_pct": "{:+.0f}%"}) + "\n")


def section_auction(out: list[str]):
    r = _load("auction")
    if r is None:
        return
    viz.auction_tradeoff(r["summary"], FIG / "auction_tradeoff.png")
    out.append("## 7. Auction study: reserve prices\n")
    out.append("60 advertisers, 100K opportunities/day, 15 segments; policies fit on day 0, evaluated on 14 held-out days "
               "(paired by day).\n")
    out.append("![](figures/auction_tradeoff.png)\n")
    s = r["summary"]
    out.append(md(s[["format", "policy", "rpm", "rpm_lift_pct", "rpm_lift_ci_low_pct", "rpm_lift_ci_high_pct",
                     "offline_predicted_lift_pct", "fill_rate", "advertiser_surplus_per_1k", "welfare_per_1k",
                     "avg_winner_quality", "budget_utilization"]],
                  {"rpm": "${:.3f}", "rpm_lift_pct": "{:+.2f}%", "rpm_lift_ci_low_pct": "{:+.2f}%",
                   "rpm_lift_ci_high_pct": "{:+.2f}%", "offline_predicted_lift_pct": "{:+.2f}%", "fill_rate": "{:.1%}",
                   "advertiser_surplus_per_1k": "${:.2f}", "welfare_per_1k": "${:.2f}", "avg_winner_quality": "{:+.3f}",
                   "budget_utilization": "{:.1%}"}) + "\n")
    seg = r["segments"]
    best = seg[seg.policy == "myerson_segment"].sort_values("avg_bidders")
    out.append("Myerson floors by segment (thin markets first):\n")
    out.append(md(best[["segment", "avg_bidders", "revenue_lift_pct", "fill_rate"]],
                  {"avg_bidders": "{:.2f}", "revenue_lift_pct": "{:+.1f}%", "fill_rate": "{:.0%}"}) + "\n")
    adv = r["advertisers"]
    a = adv[adv.policy != "fixed_$2"].groupby(["policy", "budget_constrained"]).agg(
        median_surplus_change=("surplus_change_pct", "median"),
        share_worse_off=("surplus_change_pct", lambda x: float((x < 0).mean()))).reset_index()
    out.append("Advertiser surplus vs. status quo:\n")
    out.append(md(a, {"median_surplus_change": "{:+.1f}%", "share_worse_off": "{:.0%}"}) + "\n")


def section_viewer(out: list[str]):
    r = _load("viewer")
    if r is None:
        return
    viz.viewer(r["weekly"], FIG / "viewer_long_term.png")
    out.append("## 8. Viewer-experience guardrail experiment\n")
    out.append("3-arm user-randomized test, 6 treatment weeks + 2 post weeks; CUPED-style regression adjustment "
               "on 2 pre-period weeks. Truth from simulating every viewer under every arm.\n")
    out.append("![](figures/viewer_long_term.png)\n")
    out.append(md(r["summary"], {c: "{:+.2f}%" for c in r["summary"].columns if c.endswith("pct")}) + "\n")
    out.append(md(r["abandonment"], {"ad_break_abandon_rate": "{:.2%}"}) + "\n")
    if "coverage_check" in r:
        c = r["coverage_check"]
        out.append(f"**Interval coverage over {int(c['draws'].max())} re-randomizations** (target 90%; the weekly cells "
                   "above come from one split of the same viewers, so they move together):\n")
        out.append(md(c, {"coverage": "{:.0%}", "z_sd": "{:.2f}"}) + "\n")


def build():
    FIG.mkdir(parents=True, exist_ok=True)
    out = ["# adlift - results report\n",
           "Generated by `adlift report` from saved results. All numbers are computed, not typed.\n"]
    for f in [section_incrementality_mc, section_mmm_mc, section_criteo, section_ads_psa, section_pipeline,
              section_meridian, section_auction, section_viewer]:
        try:
            f(out)
        except Exception as e:  # keep building other sections
            out.append(f"\n> section {f.__name__} failed: {e}\n")
    (ROOT / "reports" / "REPORT.md").write_text("\n".join(out))
    print(f"wrote {ROOT / 'reports' / 'REPORT.md'}")
