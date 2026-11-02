"""Heterogeneous effects: segments, multiplicity, winner's curse, frequency.

Advertisers always ask "which audience worked best?" Three things make that
question harder than it looks:

1. **Noise.** Each segment has a fraction of the sample, so segment intervals
   are wide. Report intervals, not rankings.
2. **Multiplicity.** With 20 segments, one will look "significant" by chance at
   the 5% level. We control the false discovery rate (Benjamini-Hochberg).
3. **Winner's curse.** The segment with the largest *estimated* lift is
   selected *because* its noise happened to be positive, so its estimate is
   biased upward even if each segment's estimator is unbiased. Re-measure the
   winner on fresh data before reallocating budget to it.

Frequency is a fourth trap: users who saw more impressions are more active and
would have converted more anyway, and frequency is decided after exposure, so a
"conversion rate by frequency" chart is not a dose-response curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adlift.estimators.core import aipw_from_nuisances


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def segment_effects(t, y, nuis: dict, groups: pd.Series, estimand: str = "ATT", min_treated: int = 30,
                    alpha: float = 0.10, truth: dict | None = None) -> pd.DataFrame:
    """AIPW effect per segment from shared nuisances, with BH q-values."""
    rows = []
    g = pd.Series(groups).reset_index(drop=True)
    t = np.asarray(t)
    for lev in sorted(g.dropna().unique()):
        m = (g == lev).to_numpy()
        if (t[m] == 1).sum() < min_treated or (t[m] == 0).sum() < min_treated:
            continue
        est = aipw_from_nuisances(t, y, nuis, estimand=estimand, mask=m, alpha=alpha)
        rec = {"segment": lev, "n": int(m.sum()), "n_exposed": int((t[m] == 1).sum()),
               "estimate": est.estimate, "se": est.se, "ci_low": est.ci[0], "ci_high": est.ci[1],
               "relative": est.relative, "pvalue": est.pvalue}
        if truth is not None and lev in truth:
            rec["truth"] = truth[lev]
            rec["covered"] = est.covers(truth[lev])
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out["qvalue_bh"] = benjamini_hochberg(out["pvalue"].to_numpy())
    return out


def winners_curse(table: pd.DataFrame) -> dict:
    """Error of the top-ranked segment's estimate (needs a ``truth`` column)."""
    top = table.loc[table["estimate"].idxmax()]
    return {"segment": top["segment"], "estimate": float(top["estimate"]), "truth": float(top["truth"]),
            "overstatement": float(top["estimate"] - top["truth"]),
            "true_rank": int((table["truth"] > top["truth"]).sum() + 1)}


def naive_frequency_curve(df: pd.DataFrame, outcome: str = "converted", max_f: int = 10) -> pd.DataFrame:
    """Conversion rate by frequency minus the unexposed rate (the misleading chart)."""
    base = df.loc[df["exposed"] == 0, outcome].mean()
    g = df[df["exposed"] == 1].groupby("frequency")[outcome].agg(["mean", "size"]).reset_index()
    g = g[g["frequency"] <= max_f]
    g["naive_effect"] = g["mean"] - base
    return g.rename(columns={"size": "n"})[["frequency", "n", "naive_effect"]]


def meta_regression(effects: pd.DataFrame, features: pd.DataFrame, formula_cols: list[str]) -> pd.DataFrame:
    """Inverse-variance weighted regression of campaign lifts on campaign traits.

    With ~16 campaigns this is descriptive: coefficients are confounded with
    whatever else differs between campaigns, and degrees of freedom are few.
    """
    d = effects.merge(features, left_on="segment", right_on="campaign_id")
    X = pd.get_dummies(d[formula_cols], drop_first=True, dtype=float)
    X.insert(0, "intercept", 1.0)
    w = 1 / d["se"].to_numpy() ** 2
    Xm, y = X.to_numpy(), d["estimate"].to_numpy()
    XtW = Xm.T * w
    beta = np.linalg.solve(XtW @ Xm, XtW @ y)
    resid = y - Xm @ beta
    k = Xm.shape[1]
    phi = max(float(np.sum(w * resid**2) / max(len(y) - k, 1)), 1.0)  # allow over-dispersion
    cov = np.linalg.inv(XtW @ Xm) * phi
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame({"term": X.columns, "coef": beta, "se": se, "z": beta / se})
