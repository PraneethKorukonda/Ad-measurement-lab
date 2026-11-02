"""Monte Carlo benchmark of incrementality estimators across selection mechanisms.

For each scenario (see ``adlift.sim.campaign.SCENARIOS``) and replication, we
simulate a campaign, then estimate the ATT of exposure on conversion (observed
for everyone) and on brand awareness (survey outcome, observed for respondents
only) with five estimators. Truth is the exact ATT from the simulator's
potential outcomes. Reported: bias, RMSE, 90% CI coverage, mean CI width, and
the rejection rate (share of replications with p < 0.10), which is the false
positive rate in the null scenario and power elsewhere.

Each replication also records the winner's curse: the age segment with the
largest estimated conversion lift, and how far its estimate overshoots its
true lift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from adlift.estimators import core
from adlift.estimators.design import make_design
from adlift.estimators.hetero import segment_effects, winners_curse
from adlift.sim.campaign import (
    SCENARIOS,
    design_frame,
    scenario_config,
    simulate_campaign,
    true_effects,
)

NUMERIC = ["log_sessions", "affinity_logit", "income"]
CATEGORICAL = ["age_group", "region", "device", "prior_purchaser", "campaign_id"]
OUTCOMES = ["converted", "aware"]


def ml_matrix(df: pd.DataFrame) -> np.ndarray:
    """Raw covariates for tree models (NaN kept; categories as integer codes)."""
    return np.column_stack([
        df["log_sessions"], df["affinity_logit"], df["income"], df["prior_purchaser"],
        (df["device"] == "mobile").astype(int), df["age_group"].astype("category").cat.codes,
        df["region"].astype("category").cat.codes, df["campaign_id"].str[1:].astype(int),
    ]).astype(float)


def one_rep(scenario: str, rep: int, n_users: int = 20_000, ml: bool = True) -> list[dict]:
    data = simulate_campaign(scenario_config(scenario, n_users=n_users, seed=1000 * rep + 7))
    df = design_frame(data.users)
    D = make_design(df, numeric=NUMERIC, categorical=CATEGORICAL)
    Xml = ml_matrix(df)
    t = df["exposed"].to_numpy()
    rows = []
    for outcome in OUTCOMES:
        y = df[outcome].to_numpy(float)
        truth = data.truth[outcome]["ATT"]
        ests = {
            "difference_in_means": lambda y=y: core.difference_in_means(t, y, "ATT"),
            "regression_adjustment": lambda y=y: core.regression_adjustment(np.nan_to_num(D.X), t, y, "ATT"),
            "ipw": lambda y=y: core.ipw(np.nan_to_num(D.X), t, y, "ATT"),
            "aipw_parametric": lambda y=y: core.aipw(D.X, t, y, "ATT"),
        }
        if ml:
            ests["aipw_ml"] = lambda y=y: core.aipw(Xml, t, y, "ATT", learner="ml", folds=2, seed=rep)
        for name, f in ests.items():
            try:
                e = f()
                rows.append({"scenario": scenario, "rep": rep, "outcome": outcome, "method": name,
                             "estimate": e.estimate, "se": e.se, "truth": truth,
                             "error": e.estimate - truth, "covered": e.covers(truth),
                             "width": e.ci[1] - e.ci[0], "reject": e.pvalue < 0.10})
            except Exception as exc:  # pragma: no cover - record failures rather than crash the run
                rows.append({"scenario": scenario, "rep": rep, "outcome": outcome, "method": name,
                             "estimate": np.nan, "error": np.nan, "failed": str(exc)[:80]})
    # winner's curse on age segments (parametric AIPW nuisances, conversion)
    y = df["converted"].to_numpy(float)
    nuis = core.aipw_nuisances(D.X, t, y)
    seg_truth = {k: v["converted"]["ATT"] for k, v in true_effects(data.users, by="age_group").items()}
    tab = segment_effects(t, y, nuis, df["age_group"], truth=seg_truth)
    wc = winners_curse(tab)
    rows.append({"scenario": scenario, "rep": rep, "outcome": "converted", "method": "winners_curse_top_age_segment",
                 "estimate": wc["estimate"], "truth": wc["truth"], "error": wc["overstatement"],
                 "covered": bool(tab.loc[tab.estimate.idxmax(), "covered"]), "true_rank": wc["true_rank"],
                 "segment_coverage": float(tab["covered"].mean())})
    return rows


def run(reps: int = 100, n_users: int = 20_000, scenarios: list[str] | None = None, n_jobs: int = 2,
        ml: bool = True) -> pd.DataFrame:
    sc = scenarios or list(SCENARIOS)
    out = Parallel(n_jobs=n_jobs)(delayed(one_rep)(s, r, n_users, ml) for s in sc for r in range(reps))
    return pd.DataFrame([x for rows in out for x in rows])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["method"] != "winners_curse_top_age_segment"]
    g = d.groupby(["scenario", "outcome", "method"], sort=False)
    out = g.agg(truth=("truth", "mean"), bias=("error", "mean"),
                rmse=("error", lambda e: float(np.sqrt(np.nanmean(np.square(e))))),
                coverage=("covered", "mean"), width=("width", "mean"), reject_rate=("reject", "mean"),
                reps=("rep", "nunique")).reset_index()
    out["relative_bias"] = out["bias"] / out["truth"].replace(0, np.nan)
    return out
