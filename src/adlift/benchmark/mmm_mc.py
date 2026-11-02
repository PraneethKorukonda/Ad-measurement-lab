"""Monte Carlo evaluation of the Bayesian MMM against simulated ground truth.

For each seed we simulate three years of weekly data and fit the MMM under six
conditions, paired on the same seed where the data are shared:

    clean / plain                 low channel correlation, exogenous spend
    correlated / plain            channel budgets move together (rho = 0.9)
    endogenous / plain            search spend chases unobserved demand
    endogenous / dark test        + a go-dark geo experiment on search
    endogenous / dark + heavy-up  + a 2x spend heavy-up geo experiment on search
    endogenous / query control    no experiments; search query volume as a control
                                  (back-door adjustment for demand, Chen et al. 2018)

Metrics per channel: bias, RMSE and 90% credible-interval coverage for ROAS and
marginal ROAS, interval width, and the share of the truly achievable budget
reallocation gain that the model's recommendation captures.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from adlift.mmm.model import LiftTest, fit_mmm, prepare
from adlift.mmm.optimize import recommend, score_against_truth
from adlift.sim.mmm import CHANNELS, MMMSimConfig, simulate_mmm

CONTROLS = ["price_index", "promo", "holiday"]
EXPERIMENTS = (("search", 110, 8, 0.0), ("search", 60, 10, 2.0))

CONDITIONS = [
    ("clean", dict(rho=0.2, endogeneity=0.0), 0),
    ("correlated", dict(rho=0.9, endogeneity=0.0), 0),
    ("endogenous", dict(rho=0.2, endogeneity=1.0), 0),
    ("endogenous", dict(rho=0.2, endogeneity=1.0), 1),
    ("endogenous", dict(rho=0.2, endogeneity=1.0), 2),
    ("endogenous", dict(rho=0.2, endogeneity=1.0), "query"),
]
CAL_LABEL = {0: "none", 1: "dark test", 2: "dark + heavy-up", "query": "query-volume control"}


def run_one(seed: int, draws: int = 500, tune: int = 500) -> list[dict]:
    rows = []
    cache = {}
    for scenario, kw, n_tests in CONDITIONS:
        key = (scenario, tuple(sorted(kw.items())))
        if key not in cache:
            cache[key] = simulate_mmm(MMMSimConfig(seed=seed, experiments=EXPERIMENTS, **kw))
        d = cache[key]
        if n_tests == "query":
            tests, controls = [], CONTROLS + ["query_volume"]
        else:
            tests, controls = [LiftTest.from_experiment(e) for e in d.experiments[:n_tests]], CONTROLS
        inp = prepare(d.df, CHANNELS, controls, spend_history=d.spend_history, lift_tests=tests)
        t0 = time.time()
        fit = fit_mmm(inp, draws=draws, tune=tune, chains=2, seed=seed, cache_key=f"mc{n_tests}")
        roas, mroas = fit.roas(), fit.mroas()
        rec = recommend(fit, seed=seed)
        sc = score_against_truth(rec, d.truth["params"])
        for j, c in enumerate(CHANNELS):
            for metric, draws_ in [("roas", roas[:, j]), ("mroas", mroas[:, j])]:
                truth = d.truth[metric][c]
                lo, hi = np.quantile(draws_, [0.05, 0.95])
                rows.append({"seed": seed, "scenario": scenario, "calibration": CAL_LABEL[n_tests],
                             "channel": c, "metric": metric, "estimate": float(draws_.mean()),
                             "truth": truth, "error": float(draws_.mean() - truth),
                             "covered": bool(lo <= truth <= hi), "width": float(hi - lo),
                             "share_of_achievable_gain": sc["share_of_achievable_gain"],
                             "max_rhat": fit.diagnostics["max_rhat"],
                             "divergences": fit.diagnostics["divergences"],
                             "fit_seconds": time.time() - t0})
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["scenario", "calibration", "metric", "channel"], sort=False)
    out = g.agg(truth=("truth", "mean"), bias=("error", "mean"),
                rmse=("error", lambda e: float(np.sqrt(np.mean(np.square(e))))),
                coverage=("covered", "mean"), width=("width", "mean"), n=("seed", "nunique")).reset_index()
    return out


def run(n_seeds: int = 40, draws: int = 500, tune: int = 500, out: str | None = None) -> pd.DataFrame:
    rows = []
    for s in range(n_seeds):
        rows += run_one(s, draws, tune)
        if out:
            pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)
