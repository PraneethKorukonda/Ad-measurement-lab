"""Ads vs. PSA: lift, activity bias, and frequency on a real randomized ad test.

Users were split into an ad arm (shown the advertiser's ads) and a PSA arm
(shown public-service announcements in the same slots: a "ghost" control that
keeps exposure opportunities identical). This design gives two things:

1. **The causal lift**: ad-arm minus PSA-arm conversion.
2. **A built-in placebo for selection bias.** PSAs cannot sell the product,
   yet PSA users who saw more PSAs convert far more. That gradient is pure
   *activity bias* (Lewis, Rao & Reiley 2011): heavy users see more ads *and*
   buy more. A naive "users who saw more ads converted more" analysis on the ad
   arm mostly measures this.

Because PSAs fill the same slots, ``total_ads`` should be distributed the same
in both arms (we test this). If so, comparing ad vs. PSA *within* bins of
``total_ads`` is a valid conditional contrast, and gives an honest frequency
curve. Caveat: the dataset's provenance is undocumented, so randomization is
an assumption we check, not a fact we know.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from adlift.data.fetch import path
from adlift.estimators import core
from adlift.estimators.hetero import benjamini_hochberg
from adlift.pipeline.runner import connect, query

FILE = "marketing_ab.csv.gz"
BINS = ["1-5", "6-15", "16-30", "31-60", "61-120", "121+"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _contrast(df: pd.DataFrame) -> dict:
    e = core.difference_in_means(df["ad"].to_numpy(), df["converted"].to_numpy(float), "ATE")
    return {"n": len(df), "n_psa": int((df["ad"] == 0).sum()), "psa_rate": e.baseline,
            "ad_rate": e.baseline + e.estimate, "lift": e.estimate, "se": e.se, "ci_low": e.ci[0],
            "ci_high": e.ci[1], "relative": e.relative, "relative_se": e.relative_se, "pvalue": e.pvalue}


def run() -> dict:
    p = str(path(FILE))
    con = connect()
    quality = query(con, "ads_psa/01_quality.sql", path=p)
    df = query(con, "ads_psa/02_users.sql", path=p)
    con.close()
    out: dict = {"quality": quality}

    out["overall"] = pd.DataFrame([_contrast(df)])

    # Is total_ads unaffected by arm? (needed for within-bin contrasts)
    ks = stats.ks_2samp(df.loc[df.ad == 1, "total_ads"], df.loc[df.ad == 0, "total_ads"])
    day_tab = pd.crosstab(df["ad"], df["most_ads_day"])
    shares = pd.crosstab(df["most_ads_day"], df["ad"], normalize="columns").reindex(DAYS)
    shares.columns = ["psa_share", "ad_share"]
    shares["difference_pp"] = 100 * (shares["ad_share"] - shares["psa_share"])
    out["day_mix"] = shares.reset_index()
    out["balance"] = pd.DataFrame([
        {"check": "total_ads distribution (KS test)", "statistic": ks.statistic, "pvalue": ks.pvalue},
        {"check": "most_ads_day distribution (chi-square)", "statistic": stats.chi2_contingency(day_tab)[0],
         "pvalue": stats.chi2_contingency(day_tab)[1]},
        {"check": "mean total_ads ad - psa", "statistic": df.loc[df.ad == 1, "total_ads"].mean()
         - df.loc[df.ad == 0, "total_ads"].mean(), "pvalue": stats.ttest_ind(
             df.loc[df.ad == 1, "total_ads"], df.loc[df.ad == 0, "total_ads"], equal_var=False).pvalue},
    ])

    # Sensitivity: the arms differ in day/hour mix (see day_mix). Re-estimate lift
    # adjusting for when users were active; if the arms were served over different
    # calendar periods, this is the first thing a reviewer will ask about.
    from adlift.estimators.design import make_design

    sens = [{"adjustment": "none", **_contrast(df)}]
    for label, cats in [("day + hour", ["most_ads_day", "most_ads_hour"]),
                        ("day + hour + exposure bin", ["most_ads_day", "most_ads_hour", "ads_bin"])]:
        D = make_design(df, categorical=cats)
        e = core.regression_adjustment(D.X, df["ad"].to_numpy(), df["converted"].to_numpy(float), "ATE",
                                       outcome_model="ols")
        sens.append({"adjustment": label, "n": len(df), "lift": e.estimate, "se": e.se, "ci_low": e.ci[0],
                     "ci_high": e.ci[1], "relative": e.relative, "pvalue": e.pvalue})
    out["sensitivity"] = pd.DataFrame(sens)[["adjustment", "lift", "se", "ci_low", "ci_high", "relative", "pvalue"]]

    # Activity bias: conversion vs. exposure count, separately by arm
    bins = []
    for b, g in df.groupby("ads_bin"):
        rec = {"ads_bin": b, **_contrast(g)}
        bins.append(rec)
    by_bin = pd.DataFrame(bins)
    by_bin = by_bin.assign(_o=by_bin["ads_bin"].map(BINS.index)).sort_values("_o").drop(columns="_o")
    by_bin["qvalue_bh"] = benjamini_hochberg(by_bin["pvalue"].to_numpy())
    out["by_exposure_bin"] = by_bin

    # Naive observational "effect of heavy exposure" inside each arm (median split)
    med = df["total_ads"].median()
    naive = []
    for arm, g in df.groupby("ad"):
        hi = (g["total_ads"] > med).astype(int).to_numpy()
        e = core.difference_in_means(hi, g["converted"].to_numpy(float), "ATE")
        naive.append({"arm": "ad" if arm else "psa", "heavy_minus_light": e.estimate, "se": e.se,
                      "ci_low": e.ci[0], "ci_high": e.ci[1], "light_rate": e.baseline})
    naive = pd.DataFrame(naive)
    ad_row, psa_row = naive.set_index("arm").loc["ad"], naive.set_index("arm").loc["psa"]
    out["activity_bias"] = naive
    out["activity_bias_share"] = float(psa_row["heavy_minus_light"] / ad_row["heavy_minus_light"])

    # Lift by day with most ads (multiplicity-adjusted)
    days = []
    for dname, g in df.groupby("most_ads_day"):
        days.append({"most_ads_day": dname, **_contrast(g)})
    by_day = pd.DataFrame(days)
    by_day["order"] = by_day["most_ads_day"].map({d: i for i, d in enumerate(DAYS)})
    by_day = by_day.sort_values("order").drop(columns="order")
    by_day["qvalue_bh"] = benjamini_hochberg(by_day["pvalue"].to_numpy())
    out["by_day"] = by_day

    # Heterogeneity test across exposure bins: Cochran's Q on lifts
    w = 1 / by_bin["se"] ** 2
    pooled = np.sum(w * by_bin["lift"]) / w.sum()
    Q = float(np.sum(w * (by_bin["lift"] - pooled) ** 2))
    out["bin_heterogeneity"] = {"Q": Q, "df": len(by_bin) - 1, "pvalue": float(stats.chi2.sf(Q, len(by_bin) - 1))}
    out["n_users"] = len(df)
    return out
