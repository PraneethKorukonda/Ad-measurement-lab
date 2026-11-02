"""Pre-estimation diagnostics: missingness, covariate balance, overlap, survey response.

These run before any effect is estimated and decide which estimates are
believable. A typical decision chain:

* Missingness: is a covariate missing at random with respect to exposure? If
  missingness differs by arm, a missing-indicator must enter the models.
* Balance (standardized mean differences, SMD): how different are exposed and
  unexposed users? |SMD| > 0.1 is the conventional "imbalanced" flag; after
  weighting, all should fall below it.
* Overlap: are there exposed users with no comparable unexposed users? If many
  propensities sit near 0 or 1, IPW variance explodes and RA extrapolates.
* Survey response: do respondents differ from non-respondents, and does
  response depend on exposure? If so, complete-case brand lift is biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def missingness_report(df: pd.DataFrame, columns: list[str], by: list[str]) -> pd.DataFrame:
    """Missing rate per column overall and by group, with a chi-square test of independence."""
    rows = []
    for c in columns:
        miss = df[c].isna()
        rec = {"column": c, "missing_rate": float(miss.mean())}
        for g in by:
            tab = pd.crosstab(df[g], miss)
            for lev, r in df.groupby(g)[c].apply(lambda s: s.isna().mean()).items():
                rec[f"missing_rate[{g}={lev}]"] = float(r)
            if tab.shape == (tab.shape[0], 2) and tab.shape[0] > 1 and miss.any():
                rec[f"chi2_p[{g}]"] = float(stats.chi2_contingency(tab)[1])
        rows.append(rec)
    return pd.DataFrame(rows)


def smd(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float:
    """Standardized mean difference with (optional) weights; pooled unweighted SD in denominator."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    x, t = x[ok], np.asarray(t)[ok]
    w = np.ones_like(x) if w is None else np.asarray(w, float)[ok]
    m1 = np.average(x[t == 1], weights=w[t == 1])
    m0 = np.average(x[t == 0], weights=w[t == 0])
    sd = np.sqrt((x[t == 1].var() + x[t == 0].var()) / 2)
    return float((m1 - m0) / sd) if sd > 0 else 0.0


def balance_table(X: np.ndarray, names: list[str], t: np.ndarray, weights: np.ndarray | None = None,
                  threshold: float = 0.1) -> pd.DataFrame:
    rows = []
    for j, n in enumerate(names):
        before = smd(X[:, j], t)
        rec = {"covariate": n, "smd_unweighted": before}
        if weights is not None:
            rec["smd_weighted"] = smd(X[:, j], t, weights)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["imbalanced_before"] = out["smd_unweighted"].abs() > threshold
    if weights is not None:
        out["imbalanced_after"] = out["smd_weighted"].abs() > threshold
    return out.sort_values("smd_unweighted", key=np.abs, ascending=False).reset_index(drop=True)


def att_weights(e: np.ndarray, t: np.ndarray, clip: float = 0.01) -> np.ndarray:
    e = np.clip(e, clip, 1 - clip)
    return np.where(t == 1, 1.0, e / (1 - e))


def overlap_summary(e: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    rows = []
    for arm in [1, 0]:
        v = e[t == arm]
        rows.append({"arm": "exposed" if arm else "unexposed", "n": len(v),
                     **{f"p{int(100 * k)}": float(np.quantile(v, k)) for k in q},
                     "share_below_0.05": float(np.mean(v < 0.05)), "share_above_0.95": float(np.mean(v > 0.95))})
    return pd.DataFrame(rows)


def survey_response_diagnostics(df: pd.DataFrame, covariates: list[str]) -> dict:
    """Response rate by arm and covariate differences between respondents and non-respondents."""
    inv = df[df["surveyed"] == 1]
    rr = inv.groupby("exposed")["responded"].mean()
    tab = pd.crosstab(inv["exposed"], inv["responded"])
    p = float(stats.chi2_contingency(tab)[1]) if tab.shape == (2, 2) else np.nan
    diffs = pd.DataFrame([{"covariate": c, "smd_respondent_vs_not": smd(inv[c].to_numpy(float), inv["responded"].to_numpy())}
                          for c in covariates])
    return {"response_rate_unexposed": float(rr.get(0, np.nan)), "response_rate_exposed": float(rr.get(1, np.nan)),
            "response_depends_on_exposure_p": p, "respondent_differences": diffs}


def srm_test(n_treat: int, n_control: int, expected_treat_share: float) -> dict:
    """Sample ratio mismatch: is the observed split consistent with the design?"""
    n = n_treat + n_control
    chi2, p = stats.chisquare([n_treat, n_control], [n * expected_treat_share, n * (1 - expected_treat_share)])
    return {"n_treat": n_treat, "n_control": n_control, "observed_share": n_treat / n,
            "expected_share": expected_treat_share, "chi2": float(chi2), "p_value": float(p)}
