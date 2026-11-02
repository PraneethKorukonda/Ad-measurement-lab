"""Criteo: can observational methods recover a randomized ad effect? (LaLonde for ads)

LaLonde (1986) took a job-training experiment, threw away the randomized
control group, replaced it with survey respondents, and showed that standard
observational estimators could not recover the experimental answer. This study
does the same for display advertising, using Criteo's 13.98M-user randomized
incrementality test.

1. **Experimental benchmark.** Users were randomized 85/15 into a treatment arm
   (eligible for ads) and a control arm (never shown ads). Only ~3.6% of the
   treatment arm was actually exposed. The intent-to-treat (ITT) effect is
   diluted; the effect *on the exposed* (TOT) is ITT / exposure rate, the Wald
   estimator, valid under randomization, one-sided noncompliance and the
   exclusion restriction.

2. **Observational analysis.** Discard the control arm, as an advertiser without
   a holdout would. Inside the treatment arm compare exposed with unexposed
   users, adjusting for the 12 pre-treatment features with regression,
   propensity weighting and doubly robust estimators. The estimand (ATT of
   exposure among the exposed) is the same as the TOT above, so the RCT answer
   is the yardstick.

3. **Heterogeneity.** Using the randomized contrast, estimate ITT by quintile of
   a cross-fitted baseline-conversion score: do ads work better for users who
   were already likely to convert?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from adlift.data.fetch import path
from adlift.estimators import core
from adlift.estimators.hetero import benjamini_hochberg
from adlift.pipeline import diagnostics as dg
from adlift.pipeline.runner import connect, query

FILE = "criteo-uplift-v2.1.csv.gz"
FEATURES = [f"f{i}" for i in range(12)]


def run(sample_n: int = 2_000_000, seed: int = 7, ml: bool = True) -> dict:
    p = str(path(FILE))
    con = connect()
    quality = query(con, "criteo/01_quality.sql", path=p)
    arms = query(con, "criteo/02_arm_summary.sql", path=p)
    full = query(con, "criteo/03_extract.sql", path=p)
    z, d = full["z"].to_numpy(), full["d"].to_numpy()

    out: dict = {"quality": quality, "arms": arms}
    n_t, n_c = int(z.sum()), int(len(z) - z.sum())
    out["srm"] = dg.srm_test(n_t, n_c, 0.85)

    bench = []
    for y in ["visit", "conversion"]:
        yy = full[y].to_numpy()
        itt = core.difference_in_means(z, yy, "ITT")
        tot = core.wald_tot(z, d, yy)
        bench.append({"outcome": y, "itt": itt.estimate, "itt_se": itt.se, "itt_relative": itt.relative,
                      "tot": tot.estimate, "tot_se": tot.se, "tot_ci_low": tot.ci[0], "tot_ci_high": tot.ci[1],
                      "tot_relative": tot.relative, "exposure_rate_treated": tot.diagnostics["compliance"]})
    out["benchmark"] = pd.DataFrame(bench)
    del full

    s = query(con, "criteo/04_treated_sample.sql", path=p, n=sample_n, seed=seed)
    con.close()
    X = s[FEATURES].to_numpy(float)
    t = s["exposure"].to_numpy()
    Xs = (X - X.mean(0)) / X.std(0)

    # overlap and balance
    e_hat = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                           min_samples_leaf=200, random_state=0).fit(X, t).predict_proba(X)[:, 1]
    out["overlap"] = dg.overlap_summary(e_hat, t)
    out["balance"] = dg.balance_table(Xs, FEATURES, t, weights=dg.att_weights(e_hat, t))

    rows = []
    for y in ["visit", "conversion"]:
        yy = s[y].to_numpy(float)
        truth = out["benchmark"].set_index("outcome").loc[y]
        ests = {
            "naive (exposed - unexposed)": core.difference_in_means(t, yy, "ATT"),
            "regression adjustment": core.regression_adjustment(Xs, t, yy, "ATT"),
            "IPW (logistic)": core.ipw(Xs, t, yy, "ATT"),
            "AIPW (logistic)": core.aipw(Xs, t, yy, "ATT"),
        }
        if ml:
            ests["AIPW (cross-fit GBM)"] = core.aipw(X, t, yy, "ATT", learner="ml", folds=2, seed=seed)
        for name, e in ests.items():
            rows.append({"outcome": y, "method": name, "estimate": e.estimate, "se": e.se,
                         "ci_low": e.ci[0], "ci_high": e.ci[1], "relative": e.relative,
                         "rct_tot": truth["tot"], "rct_tot_se": truth["tot_se"],
                         "bias_vs_rct": e.estimate - truth["tot"],
                         "ratio_to_rct": e.estimate / truth["tot"],
                         "z_vs_rct": (e.estimate - truth["tot"]) / np.sqrt(e.se**2 + truth["tot_se"] ** 2),
                         "weight_sum_ratio": e.diagnostics.get("weight_sum_ratio", np.nan),
                         "ess_control": e.diagnostics.get("ess_control", np.nan)})
    out["observational"] = pd.DataFrame(rows)
    out["sample_n"] = len(s)
    out["sample_exposed"] = int(t.sum())
    return out


def heterogeneity(n_rows: int = 3_000_000, seed: int = 11, outcome: str = "visit", groups: int = 5) -> pd.DataFrame:
    """ITT by quintile of a cross-fitted baseline score (randomized contrast within groups).

    The score predicts the outcome from features using *control-arm* users only,
    so it measures baseline propensity, unaffected by ads. Cross-fitting (score
    each half with a model trained on the other half) prevents the control
    users' own outcomes from leaking into their group assignment.
    """
    p = str(path(FILE))
    con = connect()
    s = con.execute(f"SELECT * FROM read_csv_auto('{p}') USING SAMPLE reservoir({n_rows} ROWS) REPEATABLE ({seed})").df()
    con.close()
    X, z, y = s[FEATURES].to_numpy(float), s["treatment"].to_numpy(), s[outcome].to_numpy(float)
    rng = np.random.default_rng(seed)
    half = rng.random(len(s)) < 0.5
    score = np.empty(len(s))
    for a, b in [(half, ~half), (~half, half)]:
        tr = a & (z == 0)
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, min_samples_leaf=200, random_state=0)
        score[b] = m.fit(X[tr], y[tr]).predict_proba(X[b])[:, 1]
    # rank first: rare outcomes give many tied scores, which would collapse quantile edges
    q = pd.qcut(pd.Series(score).rank(method="first"), groups, labels=False).to_numpy()
    rows = []
    for g in range(groups):
        m = q == g
        e = core.difference_in_means(z[m], y[m], "ITT")
        rows.append({"score_quintile": g + 1, "n": int(m.sum()), "baseline_rate": e.baseline, "itt": e.estimate,
                     "se": e.se, "ci_low": e.ci[0], "ci_high": e.ci[1], "relative": e.relative, "pvalue": e.pvalue})
    out = pd.DataFrame(rows)
    out["qvalue_bh"] = benjamini_hochberg(out["pvalue"].to_numpy())
    # test of heterogeneity: top vs bottom quintile
    top, bot = out.iloc[-1], out.iloc[0]
    diff = top["itt"] - bot["itt"]
    se = np.sqrt(top["se"] ** 2 + bot["se"] ** 2)
    out.attrs["top_minus_bottom"] = (diff, se)
    out.attrs["outcome"] = outcome
    return out
