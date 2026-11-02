"""Command-line entry point: ``adlift <command>``.

Every command writes its outputs to ``reports/data`` so the report and figures
can be rebuilt without re-running expensive steps.

    adlift fetch               download and verify public datasets
    adlift pipeline            simulate raw campaign logs, run the SQL pipeline, diagnostics
    adlift criteo              RCT benchmark vs. observational estimators (real data)
    adlift ads-psa             lift, activity bias and frequency (real data)
    adlift meridian            Bayesian MMM on Google Meridian's demo data
    adlift auction             reserve-price study (revenue, fill, advertisers)
    adlift viewer              viewer-experience guardrail experiment
    adlift mc-incrementality   Monte Carlo benchmark of incrementality estimators
    adlift mc-mmm              Monte Carlo benchmark of the MMM against ground truth
    adlift report              figures + markdown report from saved results
    adlift all                 everything above, in order
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "data"


def _save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load(name: str):
    with open(OUT / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def cmd_fetch(a):
    from adlift.data.fetch import fetch

    fetch()


def cmd_pipeline(a):
    import numpy as np

    from adlift.estimators import core
    from adlift.estimators.design import make_design
    from adlift.estimators.hetero import (
        meta_regression,
        naive_frequency_curve,
        segment_effects,
    )
    from adlift.pipeline import diagnostics as dg
    from adlift.pipeline.runner import build_campaign_tables
    from adlift.sim.campaign import (
        CampaignConfig,
        simulate_campaign,
        to_raw_tables,
        true_effects,
        true_frequency_curve,
    )

    # One large "production-like" campaign flight: targeted delivery + MAR survey nonresponse
    data = simulate_campaign(CampaignConfig(n_users=a.n_users, selection="linear", nonresponse="mar", seed=a.seed))
    raw = to_raw_tables(data, inject_issues=True, seed=a.seed)
    res = build_campaign_tables(raw)
    df = res.analysis
    t = df["exposed"].to_numpy()
    num, cat = ["log_sessions", "affinity_logit", "income"], ["age_group", "region", "device", "prior_purchaser", "campaign_id"]
    D = make_design(df, numeric=num, categorical=cat)
    from adlift.benchmark.incrementality_mc import ml_matrix

    Xml = ml_matrix(df)
    e_hat = core.fit_glm("logit", np.nan_to_num(D.X), t).predict()
    out = {
        "raw_quality": res.raw_quality, "analysis_quality": res.analysis_quality,
        "campaign_summary": res.campaign_summary, "n_users": len(df),
        "missingness": dg.missingness_report(df, ["income", "aware", "intent"], by=["device", "exposed"]),
        "balance": dg.balance_table(np.nan_to_num(D.X), D.names, t, weights=dg.att_weights(e_hat, t)),
        "overlap": dg.overlap_summary(e_hat, t),
        "survey": dg.survey_response_diagnostics(df, ["log_sessions", "affinity_logit", "exposed"]),
        "truth": data.truth,
    }
    # Effects on all three outcomes with all estimators
    rows = []
    for o in ["converted", "aware", "intent"]:
        y = df[o].to_numpy(float)
        tr = data.truth[o]["ATT"]
        for name, est in [("difference_in_means", core.difference_in_means(t, y, "ATT")),
                          ("regression_adjustment", core.regression_adjustment(np.nan_to_num(D.X), t, y, "ATT")),
                          ("ipw", core.ipw(np.nan_to_num(D.X), t, y, "ATT")),
                          ("aipw_parametric", core.aipw(D.X, t, y, "ATT")),
                          ("aipw_ml", core.aipw(Xml, t, y, "ATT", learner="ml", folds=5, seed=a.seed))]:
            rows.append({"outcome": o, "method": name, "estimate": est.estimate, "se": est.se,
                         "ci_low": est.ci[0], "ci_high": est.ci[1], "relative": est.relative,
                         "truth": tr, "covered": est.covers(tr)})
    out["effects"] = rows
    # Segments (AIPW-ML nuisances on full sample), campaign meta-regression, frequency
    seg = {}
    for o in ["converted", "aware"]:
        y = df[o].to_numpy(float)
        nuis = core.aipw_nuisances(Xml, t, y, learner="ml", folds=5, seed=a.seed)
        for by in ["age_group", "device", "prior_purchaser", "campaign_id"]:
            tr = {k: v[o]["ATT"] for k, v in true_effects(data.users, by=by).items()}
            seg[(o, by)] = segment_effects(t, y, nuis, df[by], truth=tr)
    out["segments"] = seg
    camp = data.campaigns[["campaign_id", "format", "objective", "frequency_cap", "creative_score"]]
    out["meta_regression"] = {o: meta_regression(seg[(o, "campaign_id")], camp, ["format", "objective", "creative_score"])
                              for o in ["converted", "aware"]}
    out["frequency"] = {"naive": naive_frequency_curve(df, "converted"),
                        "true": true_frequency_curve(data.users, "converted")}
    # Brand-lift business metrics for the flight
    spend = float(df["media_cost_usd"].sum())
    reached = int(t.sum())
    aw = [r for r in rows if r["outcome"] == "aware" and r["method"] == "aipw_ml"][0]
    out["brand_lift_kpis"] = {"spend_usd": spend, "reached_users": reached,
                              "absolute_lift_pp": 100 * aw["estimate"],
                              "lifted_users": aw["estimate"] * reached,
                              "lifted_users_ci": (aw["ci_low"] * reached, aw["ci_high"] * reached),
                              "cost_per_lifted_user": spend / (aw["estimate"] * reached),
                              "true_lifted_users": data.truth["aware"]["ATT"] * reached}
    _save("pipeline", out)
    print(res.raw_quality.to_string())


def cmd_criteo(a):
    from adlift.studies import criteo

    r = criteo.run(sample_n=a.sample_n)
    _save("criteo", r)  # save the main study before the (slower) heterogeneity step
    r["heterogeneity"] = {o: criteo.heterogeneity(outcome=o) for o in ["visit", "conversion"]}
    _save("criteo", r)
    print(r["benchmark"].to_string())
    print(r["observational"].to_string())


def cmd_ads_psa(a):
    from adlift.studies import ads_psa

    r = ads_psa.run()
    _save("ads_psa", r)
    print(r["overall"].to_string())


def cmd_meridian(a):
    from adlift.studies import meridian

    r = meridian.run(draws=a.draws, tune=a.tune)
    r.pop("fit")
    _save("meridian", r)
    print(r["summary"].to_string())


def cmd_auction(a):
    from adlift.auction.study import run_reserve_study

    r = run_reserve_study(n_days=a.days)
    _save("auction", r)
    print(r["summary"].to_string())


def cmd_viewer(a):
    from adlift.auction.viewer import ViewerConfig, run_viewer_experiment

    r = run_viewer_experiment(ViewerConfig(n_viewers=a.n_viewers))
    _save("viewer", r)
    print(r["summary"].to_string())


def cmd_mc_incrementality(a):
    from adlift.benchmark import incrementality_mc as mc

    df = mc.run(reps=a.reps, n_users=a.n_users)
    df.to_csv(OUT / "incrementality_mc_raw.csv", index=False)
    s = mc.summarize(df)
    s.to_csv(OUT / "incrementality_mc_summary.csv", index=False)
    print(s.to_string())


def cmd_mc_mmm(a):
    from adlift.benchmark import mmm_mc

    df = mmm_mc.run(n_seeds=a.seeds, out=str(OUT / "mmm_mc_raw.csv"))
    s = mmm_mc.summarize(df)
    s.to_csv(OUT / "mmm_mc_summary.csv", index=False)
    print(s.to_string())


def cmd_report(a):
    from adlift import report

    report.build()


def main(argv=None):
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    p = argparse.ArgumentParser(prog="adlift", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(f=cmd_fetch)
    sp = sub.add_parser("pipeline")
    sp.add_argument("--n-users", type=int, default=200_000)
    sp.add_argument("--seed", type=int, default=2026)
    sp.set_defaults(f=cmd_pipeline)
    sp = sub.add_parser("criteo")
    sp.add_argument("--sample-n", type=int, default=2_000_000)
    sp.set_defaults(f=cmd_criteo)
    sub.add_parser("ads-psa").set_defaults(f=cmd_ads_psa)
    sp = sub.add_parser("meridian")
    sp.add_argument("--draws", type=int, default=1000)
    sp.add_argument("--tune", type=int, default=1500)
    sp.set_defaults(f=cmd_meridian)
    sp = sub.add_parser("auction")
    sp.add_argument("--days", type=int, default=14)
    sp.set_defaults(f=cmd_auction)
    sp = sub.add_parser("viewer")
    sp.add_argument("--n-viewers", type=int, default=60_000)
    sp.set_defaults(f=cmd_viewer)
    sp = sub.add_parser("mc-incrementality")
    sp.add_argument("--reps", type=int, default=100)
    sp.add_argument("--n-users", type=int, default=20_000)
    sp.set_defaults(f=cmd_mc_incrementality)
    sp = sub.add_parser("mc-mmm")
    sp.add_argument("--seeds", type=int, default=40)
    sp.set_defaults(f=cmd_mc_mmm)
    sub.add_parser("report").set_defaults(f=cmd_report)
    sp = sub.add_parser("all")
    sp.set_defaults(f=None)
    a = p.parse_args(argv)
    if a.cmd == "all":
        for c in ["fetch", "pipeline", "criteo", "ads-psa", "meridian", "auction", "viewer",
                  "mc-incrementality", "mc-mmm", "report"]:
            t0 = time.time()
            main([c])
            print(f"[{c}] {time.time() - t0:.0f}s")
        return
    a.f(a)


if __name__ == "__main__":
    main()
