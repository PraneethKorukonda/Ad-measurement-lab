"""Bayesian MMM on Google Meridian's public demo dataset.

Meridian (Google's open-source MMM) ships a simulated 40-geo, 156-week dataset
with five paid channels. We aggregate it to national weekly data in SQL, fit
our PyMC MMM, validate it on a 13-week time holdout, and produce the advertiser
outputs: ROAS and marginal ROAS with credible intervals, and a budget
reallocation with its uncertainty.

What this study can and cannot show: the CSV does not include the generating
parameters, so there is no ground truth to score against here (that is what
the simulation benchmark is for). The value of this study is the applied
workflow on data with realistic messiness: correlated channels (r up to 0.7),
a channel that is dark in most geo-weeks, and controls of unknown provenance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adlift.data.fetch import path
from adlift.mmm.model import fit_mmm, prepare
from adlift.mmm.optimize import recommend
from adlift.pipeline.runner import connect, query

CHANNELS = [f"channel{i}" for i in range(5)]
CONTROLS = ["competitor_sales", "sentiment", "promo", "organic_impressions"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = connect()
    p = str(path("meridian_geo_all_channels.csv"))
    quality = query(con, "mmm/00_meridian_quality.sql", path=p)
    df = query(con, "mmm/01_meridian_national_from_geo.sql", path=p)
    con.close()
    df["organic_impressions"] = np.log1p(df["organic_impressions"])
    return df, quality


def run(draws: int = 1000, tune: int = 1000, holdout_weeks: int = 13, seed: int = 0) -> dict:
    df, quality = load()
    train = df.iloc[:-holdout_weeks].reset_index(drop=True)

    # 1) out-of-sample check: fit on the first 143 weeks, predict the last 13
    inp_tr = prepare(train, CHANNELS, CONTROLS)
    fit_tr = fit_mmm(inp_tr, draws=draws, tune=tune, chains=2, seed=seed, target_accept=0.95)
    inp_all = prepare(df, CHANNELS, CONTROLS, scaling=inp_tr.scaling)
    pred = fit_tr.predict(inp_all).mean(axis=0)
    y = df["revenue"].to_numpy()
    T0 = len(train)

    def mape(a, b):
        return float(np.mean(np.abs(a - b) / a))

    naive = np.full(holdout_weeks, train["revenue"].iloc[-13:].mean())
    validation = {
        "train_mape": mape(y[:T0], pred[:T0]),
        "train_r2": float(1 - np.sum((y[:T0] - pred[:T0]) ** 2) / np.sum((y[:T0] - y[:T0].mean()) ** 2)),
        "holdout_mape": mape(y[T0:], pred[T0:]),
        "holdout_mape_naive_last13_mean": mape(y[T0:], naive),
    }

    # 2) final fit on all weeks for reporting
    inp = prepare(df, CHANNELS, CONTROLS)
    fit = fit_mmm(inp, draws=draws, tune=tune, chains=2, seed=seed, target_accept=0.95)
    summary = fit.summary()
    summary["spend_share"] = summary["spend"] / summary["spend"].sum()
    contrib = fit.contributions().mean(axis=0).sum(axis=0)
    rec = recommend(fit, seed=seed)
    alloc = pd.DataFrame({"channel": CHANNELS, "current_weekly": rec["current"],
                          "recommended_weekly": rec["recommended"]})
    alloc["change_pct"] = 100 * (alloc["recommended_weekly"] / alloc["current_weekly"] - 1)
    curves = []
    grid = np.linspace(0, 2.0, 41)
    for j, c in enumerate(CHANNELS):
        for g in grid:
            s = rec["current"].copy()
            s[j] = g * rec["current"][j]
            r = fit.weekly_response(s)[:, j]
            curves.append({"channel": c, "spend_multiple": g, "weekly_spend": s[j], "mean": r.mean(),
                           "low": np.quantile(r, 0.05), "high": np.quantile(r, 0.95)})
    return {
        "quality": quality, "data": df, "fit_diagnostics": fit.diagnostics,
        "validation": validation, "holdout_pred": pred, "summary": summary,
        "media_share_of_revenue": float(contrib.sum() / y.sum()),
        "allocation": alloc, "reallocation_gain": {"mean": rec["gain_mean"], "ci": rec["gain_ci"],
                                                   "p_positive": rec["p_gain_positive"],
                                                   "pct_of_revenue": rec["gain_mean"] / y.mean()},
        "curves": pd.DataFrame(curves), "fit": fit,
    }
