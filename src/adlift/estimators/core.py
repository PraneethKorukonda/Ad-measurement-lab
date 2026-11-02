"""Incrementality estimators: difference in means, regression adjustment, IPW, AIPW, Wald.

Why several estimators? Each fails differently, and knowing *how* is the point.

* **Difference in means** is unbiased only when exposure is randomized (a
  holdout or ghost-ads test). Under targeting it measures who was targeted,
  not what the ad did.
* **Regression adjustment (RA, g-computation)** models the outcome. It removes
  bias from observed confounders if the outcome model is right, and silently
  extrapolates when treated and control users do not overlap.
* **Inverse propensity weighting (IPW)** models exposure instead. It is right
  if the propensity model is right, but a few near-0 or near-1 propensities can
  make its variance explode.
* **Augmented IPW (AIPW, doubly robust)** combines both and is consistent if
  *either* model is right. With cross-fitted machine-learning nuisances it also
  attains the semiparametric efficiency bound, and its influence function gives
  valid standard errors without bootstrapping.

None of them can fix *unobserved* confounding. Only randomization can.

Brand-lift outcomes add a second missing-data problem: awareness and purchase
intent come from surveys, so the outcome is missing for everyone who was not
surveyed or did not respond. All estimators accept ``y`` with NaN for missing
outcomes. Under missing-at-random given (X, exposure), RA and AIPW use a
response model ``r(x, t) = P(responded | x, t)`` and weight respondents by
``1 / r``; the naive complete-case comparison is biased whenever response
depends on exposure or on covariates that drive the outcome.

Standard errors
---------------
Every SE is computed from a per-unit influence function. For RA and IPW, whose
nuisance models are estimated parametrically, the influence function includes
the M-estimation correction for the estimated coefficients (numerically
differentiated), so intervals account for the fact that the models were fit.
AIPW needs no such correction because its score is Neyman-orthogonal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from adlift.results import EffectEstimate, from_influence

# ---------------------------------------------------------------------------
# Parametric nuisance models with influence functions
# ---------------------------------------------------------------------------


@dataclass
class GLM:
    """Logistic or linear regression fit on a subset ``mask`` of units.

    Stores the per-unit influence function of the coefficients, scaled to the
    full sample of size n so that corrections can be added to any estimator's
    influence function directly.
    """

    kind: str  # "logit" or "ols"
    Xd: np.ndarray  # design with intercept, all n units
    coef: np.ndarray
    influence: np.ndarray  # (n, k)

    def predict(self, coef: np.ndarray | None = None, Xd: np.ndarray | None = None) -> np.ndarray:
        b = self.coef if coef is None else coef
        X = self.Xd if Xd is None else Xd
        eta = X @ b
        return expit(eta) if self.kind == "logit" else eta


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def fit_glm(kind: str, X: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None,
            ridge: float = 1e-6, max_iter: int = 200) -> GLM:
    """Fit by Newton-Raphson (logit) or least squares (ols) on ``mask`` units."""
    Xd = _add_intercept(X)
    n, k = Xd.shape
    m = np.ones(n, bool) if mask is None else mask.astype(bool)
    Xm, zm = Xd[m], np.nan_to_num(target[m].astype(float))
    pen = ridge * len(Xm) * np.eye(k)
    pen[0, 0] = 0.0
    if kind == "ols":
        H = Xm.T @ Xm + pen
        coef = np.linalg.solve(H, Xm.T @ zm)
        resid = np.where(m, np.nan_to_num(target.astype(float)) - Xd @ coef, 0.0)
        scores = Xd * resid[:, None]
        hess = H / n
    elif kind == "logit":
        coef = np.zeros(k)
        pbar = np.clip(zm.mean(), 1e-4, 1 - 1e-4)
        coef[0] = np.log(pbar / (1 - pbar))
        # Newton-Raphson with backtracking line search on the penalized log-likelihood.
        # Plain Newton can overshoot on heavy-tailed features with a rare outcome; a
        # capped step without a line search can stall far from the optimum (it did on
        # Criteo), so every step must increase the objective, and we stop on the gradient.
        def objective(b):
            eta = Xm @ b
            return float(np.sum(zm * eta - np.logaddexp(0.0, eta)) - 0.5 * b @ pen @ b)

        obj = objective(coef)
        for _ in range(max_iter):
            p = expit(Xm @ coef)
            g = Xm.T @ (zm - p) - pen @ coef
            if np.max(np.abs(g)) / len(Xm) < 1e-10:
                break
            H = (Xm * (p * (1 - p))[:, None]).T @ Xm + pen + 1e-12 * len(Xm) * np.eye(k)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            lr = 1.0
            while lr > 1e-6:
                cand = coef + lr * step
                new = objective(cand)
                if new >= obj - 1e-12:
                    break
                lr *= 0.5
            coef, obj = cand, new
            if np.max(np.abs(lr * step)) < 1e-10:
                break
        p_all = expit(Xd @ coef)
        resid = np.where(m, np.nan_to_num(target.astype(float)) - p_all, 0.0)
        scores = Xd * resid[:, None]
        pm = expit(Xm @ coef)
        hess = ((Xm * (pm * (1 - pm))[:, None]).T @ Xm + pen) / n
    else:  # pragma: no cover
        raise ValueError(kind)
    try:
        influence = np.linalg.solve(hess, scores.T).T
    except np.linalg.LinAlgError:
        influence = (np.linalg.pinv(hess) @ scores.T).T
    return GLM(kind=kind, Xd=Xd, coef=coef, influence=influence)


def _corrected_influence(psi: Callable[[list[np.ndarray]], np.ndarray],
                         models: list[GLM], rel_step: float = 1e-5) -> np.ndarray:
    """psi_i + sum_m (d mean psi / d coef_m) . IF_coef_m,i  (stacked M-estimation)."""
    base_coefs = [m.coef for m in models]
    out = psi(base_coefs)
    for j, mdl in enumerate(models):
        k = len(mdl.coef)
        grad = np.zeros(k)
        for a in range(k):
            h = rel_step * max(1.0, abs(mdl.coef[a]))
            up = [c.copy() for c in base_coefs]
            dn = [c.copy() for c in base_coefs]
            up[j][a] += h
            dn[j][a] -= h
            grad[a] = (psi(up).mean() - psi(dn).mean()) / (2 * h)
        out = out + mdl.influence @ grad
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prep(t: np.ndarray, y: np.ndarray):
    t = np.asarray(t).astype(float)
    y = np.asarray(y, dtype=float)
    R = ~np.isnan(y)
    y0 = np.where(R, y, 0.0)
    return t, y0, R.astype(float)


def _is_binary(y: np.ndarray) -> bool:
    v = y[~np.isnan(y)]
    return bool(np.isin(v, [0.0, 1.0]).all())


def _kish_ess(w: np.ndarray) -> float:
    w = w[w > 0]
    return float(w.sum() ** 2 / np.sum(w**2)) if len(w) else 0.0


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def difference_in_means(t, y, estimand: str = "ATE", alpha: float = 0.10) -> EffectEstimate:
    """Complete-case difference in means (valid under randomization with MCAR outcomes)."""
    t, y0, R = _prep(t, y)
    a1, a0 = t * R, (1 - t) * R
    m1, m0 = (a1 * y0).sum() / a1.sum(), (a0 * y0).sum() / a0.sum()
    if1 = a1 * (y0 - m1) / a1.mean()
    if0 = a0 * (y0 - m0) / a0.mean()
    return from_influence(estimand, "difference_in_means", m1 - m0, if1 - if0,
                          baseline=m0, baseline_influence=if0, alpha=alpha,
                          diagnostics={"n_treated_obs": int(a1.sum()), "n_control_obs": int(a0.sum())})


def regression_adjustment(X, t, y, estimand: str = "ATT", outcome_model: str | None = None,
                          alpha: float = 0.10) -> EffectEstimate:
    """G-computation with separate outcome models per arm.

    ATT = mean over treated of [mu1(x) - mu0(x)] when some outcomes are missing,
    or of [y - mu0(x)] when all treated outcomes are observed. Outcome models are
    fit on respondents only, which is valid if response is MAR given (x, t).
    """
    t, y0, R = _prep(t, y)
    kind = outcome_model or ("logit" if _is_binary(y) else "ols")
    complete = bool(R.all())
    m0 = fit_glm(kind, X, y0, mask=(1 - t) * R > 0)
    m1 = fit_glm(kind, X, y0, mask=t * R > 0)
    p = t.mean()

    if estimand == "ATT":
        def mu1_part(c1):
            return y0 if complete else m1.predict(c1)

        tau = float(np.sum(t * (mu1_part(m1.coef) - m0.predict())) / t.sum())
        base = float(np.sum(t * m0.predict()) / t.sum())

        def psi(cs):
            c0, c1 = cs
            return t * (mu1_part(c1) - m0.predict(c0) - tau) / p

        def psi_base(cs):
            return t * (m0.predict(cs[0]) - base) / p
    elif estimand == "ATE":
        tau = float(np.mean(m1.predict() - m0.predict()))
        base = float(np.mean(m0.predict()))

        def psi(cs):
            c0, c1 = cs
            return m1.predict(c1) - m0.predict(c0) - tau

        def psi_base(cs):
            return m0.predict(cs[0]) - base
    else:  # pragma: no cover
        raise ValueError(estimand)
    inf = _corrected_influence(psi, [m0, m1])
    inf_b = _corrected_influence(psi_base, [m0, m1])
    return from_influence(estimand, "regression_adjustment", tau, inf, baseline=base,
                          baseline_influence=inf_b, alpha=alpha)


def ipw(X, t, y, estimand: str = "ATT", clip: float = 0.01, X_response=None,
        alpha: float = 0.10) -> EffectEstimate:
    """Normalized (Hajek) inverse propensity weighting.

    Control weights for the ATT are e/(1-e): controls who look like exposed users
    count more. If outcomes are missing, respondents are additionally weighted by
    1/r(x, t), with r from a logistic response model on ``X_response`` (defaults
    to X) plus the exposure indicator.
    """
    t, y0, R = _prep(t, y)
    ps = fit_glm("logit", X, t)
    models = [ps]
    missing = not R.all()
    if missing:
        Xr = np.column_stack([X if X_response is None else X_response, t])
        rm = fit_glm("logit", Xr, R)
        models.append(rm)

    def weights(cs):
        e = np.clip(ps.predict(cs[0]), clip, 1 - clip)
        om = R / np.clip(models[1].predict(cs[1]), 0.02, 1.0) if missing else R
        return e, om

    e, om = weights([m.coef for m in models])
    if estimand == "ATT":
        w1 = t * om
        w0 = (1 - t) * om * e / (1 - e)
    else:
        w1 = t * om / e
        w0 = (1 - t) * om / (1 - e)
    mu1 = (w1 * y0).sum() / w1.sum()
    mu0 = (w0 * y0).sum() / w0.sum()
    tau = mu1 - mu0
    d1, d0 = w1.mean(), w0.mean()

    def parts(cs):
        e_, om_ = weights(cs)
        if estimand == "ATT":
            a1, a0 = t * om_, (1 - t) * om_ * e_ / (1 - e_)
        else:
            a1, a0 = t * om_ / e_, (1 - t) * om_ / (1 - e_)
        return a1 * (y0 - mu1) / d1, a0 * (y0 - mu0) / d0

    inf = _corrected_influence(lambda cs: parts(cs)[0] - parts(cs)[1], models)
    inf0 = _corrected_influence(lambda cs: parts(cs)[1], models)
    diag = {
        "ess_control": _kish_ess(w0),
        "ess_treated": _kish_ess(w1),
        "max_control_weight_share": float(w0.max() / w0.sum()),
        "share_clipped": float(np.mean((ps.predict() < clip) | (ps.predict() > 1 - clip))),
    }
    return from_influence(estimand, "ipw", tau, inf, baseline=mu0, baseline_influence=inf0,
                          alpha=alpha, diagnostics=diag)


def _ml_classifier(seed: int):
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)


def _ml_regressor(seed: int):
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=seed)


def _fit_predict(kind: str, learner: str, Xtr, ztr, Xte, seed: int):
    """Fit a nuisance on train rows, predict on test rows."""
    if len(np.unique(ztr)) < 2:
        return np.full(len(Xte), float(ztr.mean()))
    if learner == "ml":
        if kind == "logit":
            return _ml_classifier(seed).fit(Xtr, ztr).predict_proba(Xte)[:, 1]
        return _ml_regressor(seed).fit(Xtr, ztr).predict(Xte)
    g = fit_glm(kind, Xtr, ztr)
    return g.predict(Xd=_add_intercept(Xte))


def aipw_nuisances(X, t, y, learner: str = "parametric", folds: int = 2, X_response=None,
                   seed: int = 0) -> dict:
    """Fit (cross-fitted) nuisance predictions: e(x), mu0(x), mu1(x) and response r(x, t)."""
    t, y0, R = _prep(t, y)
    n = len(t)
    kind = "logit" if _is_binary(y) else "ols"
    missing = not R.all()
    Xr = np.column_stack([X if X_response is None else X_response, t])
    e, mu0, mu1, r = np.empty(n), np.empty(n), np.empty(n), np.ones(n)
    if learner == "ml" and folds > 1:
        strat = (t * 2 + R).astype(int)
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, strat))
    else:
        idx = np.arange(n)
        splits = [(idx, idx)]
    Xp = X if learner == "ml" else np.nan_to_num(X)
    Xrp = Xr if learner == "ml" else np.nan_to_num(Xr)
    for k, (tr, te) in enumerate(splits):
        s = seed + 17 * k
        e[te] = _fit_predict("logit", learner, Xp[tr], t[tr], Xp[te], s)
        c0 = tr[(t[tr] == 0) & (R[tr] == 1)]
        c1 = tr[(t[tr] == 1) & (R[tr] == 1)]
        mu0[te] = _fit_predict(kind, learner, Xp[c0], y0[c0], Xp[te], s + 1)
        mu1[te] = _fit_predict(kind, learner, Xp[c1], y0[c1], Xp[te], s + 2)
        if missing:
            r[te] = _fit_predict("logit", learner, Xrp[tr], R[tr], Xrp[te], s + 3)
    return {"e": e, "mu0": mu0, "mu1": mu1, "r": r, "learner": learner, "folds": len(splits)}


def aipw_from_nuisances(t, y, nuis: dict, estimand: str = "ATT", mask=None, clip: float = 0.01,
                        normalize: bool = True, alpha: float = 0.10) -> EffectEstimate:
    """AIPW estimate from pre-fit nuisances, optionally within a subgroup ``mask``.

    Subgroup effects reuse nuisances fit on the full sample (more data per model)
    and simply average the doubly robust scores within the subgroup.

    ``normalize`` rescales the augmentation weights within each arm so they sum to
    the size of the population they stand in for (Hajek-style). If the propensity
    model is right this changes nothing asymptotically; if it is miscalibrated,
    unnormalized weights can sum to several times the right total and amplify
    outcome-model residuals. ``weight_sum_ratio`` in the diagnostics reports the
    unnormalized sum / target: far from 1 is a red flag for the propensity model.
    """
    t, y0, R = _prep(t, y)
    m = np.ones(len(t), bool) if mask is None else np.asarray(mask, bool)
    t, y0, R = t[m], y0[m], R[m]
    e = np.clip(nuis["e"][m], clip, 1 - clip)
    mu0, mu1 = nuis["mu0"][m], nuis["mu1"][m]
    om = R / np.clip(nuis["r"][m], 0.02, 1.0)
    p = t.mean()
    if estimand == "ATT":
        odds = e / (1 - e)
        w0 = (1 - t) * odds * om
        w1 = t * om
        ratio = w0.sum() / t.sum()
        if normalize:
            w0 = w0 / ratio
            w1 = w1 * t.sum() / max(w1.sum(), 1e-12)
        a1 = t * mu1 + w1 * (y0 - mu1)
        a0 = t * mu0 + w0 * (y0 - mu0)
        m1, m0 = a1.sum() / t.sum(), a0.sum() / t.sum()
        inf1, inf0 = (a1 - t * m1) / p, (a0 - t * m0) / p
        wctrl = w0
    elif estimand == "ATE":
        w1 = t * om / e
        w0 = (1 - t) * om / (1 - e)
        ratio = w0.sum() / len(t)
        if normalize:
            w1 = w1 * len(t) / max(w1.sum(), 1e-12)
            w0 = w0 / ratio
        a1 = mu1 + w1 * (y0 - mu1)
        a0 = mu0 + w0 * (y0 - mu0)
        m1, m0 = a1.mean(), a0.mean()
        inf1, inf0 = a1 - m1, a0 - m0
        wctrl = w0
    else:  # pragma: no cover
        raise ValueError(estimand)
    diag = {
        "ess_control": _kish_ess(wctrl),
        "weight_sum_ratio": float(ratio),
        "e_treated_mean": float(e[t == 1].mean()),
        "e_control_mean": float(e[t == 0].mean()),
        "share_e_above_0.95": float(np.mean(e > 0.95)),
        "share_e_below_0.05": float(np.mean(e < 0.05)),
        "learner": nuis["learner"],
        "folds": nuis["folds"],
    }
    return from_influence(estimand, f"aipw_{nuis['learner']}", m1 - m0, inf1 - inf0, baseline=m0,
                          baseline_influence=inf0, alpha=alpha, diagnostics=diag)


def aipw(X, t, y, estimand: str = "ATT", learner: str = "parametric", folds: int = 2,
         clip: float = 0.01, X_response=None, seed: int = 0, alpha: float = 0.10) -> EffectEstimate:
    """Augmented IPW (doubly robust), optionally cross-fitted with gradient boosting.

    With ``learner="ml"`` nuisances are HistGradientBoosting models fit out of
    fold (cross-fitting), which removes the own-observation overfitting bias that
    otherwise breaks inference with flexible learners. ``learner="parametric"``
    uses logistic/linear models fit in-sample (no cross-fitting needed).
    NaNs in X are handled natively by the ML learners.
    """
    nuis = aipw_nuisances(X, t, y, learner=learner, folds=folds, X_response=X_response, seed=seed)
    return aipw_from_nuisances(t, y, nuis, estimand=estimand, clip=clip, alpha=alpha)


def wald_tot(z, d, y, alpha: float = 0.10) -> EffectEstimate:
    """Effect of exposure on the exposed under randomized assignment with noncompliance.

    In a holdout test, users assigned to the treatment arm are only *eligible*;
    many never see an ad (they did not visit, lost the auction, ...). The
    intent-to-treat (ITT) contrast is diluted by those never-exposed users. With
    one-sided noncompliance (control users cannot be exposed) and the exclusion
    restriction (assignment changes outcomes only through exposure),
    TOT = ITT_y / ITT_d, the Wald/IV estimator. This is the experimental
    benchmark that a naive "exposed vs. unexposed" comparison tries to estimate.

    ``baseline`` is E[Y(0) | complier], so ``relative`` is the lift for exposed users.
    """
    z = np.asarray(z, float)
    d = np.asarray(d, float)
    y = np.asarray(y, float)
    pz = z.mean()

    def mean_if(v, arm):
        m = (arm * v).sum() / arm.sum()
        return m, arm * (v - m) / arm.mean()

    my1, iy1 = mean_if(y, z)
    my0, iy0 = mean_if(y, 1 - z)
    md1, id1 = mean_if(d, z)
    md0, id0 = mean_if(d, 1 - z)
    b, ib = mean_if(y * (1 - d), z)  # E[Y(1-D) | Z=1] = (1-pi) E[Y0 | never-taker]
    itt_y, itt_d = my1 - my0, md1 - md0
    tot = itt_y / itt_d
    if_itt_y, if_itt_d = iy1 - iy0, id1 - id0
    inf = (if_itt_y - tot * if_itt_d) / itt_d
    base = (my0 - b) / md1
    if_base = ((iy0 - ib) - base * id1) / md1
    return from_influence("TOT", "wald_iv", tot, inf, baseline=base, baseline_influence=if_base,
                          alpha=alpha, diagnostics={"itt": itt_y, "compliance": itt_d, "p_assigned": pz})


ESTIMATORS = {
    "difference_in_means": lambda X, t, y, **k: difference_in_means(t, y, k.get("estimand", "ATT")),
    "regression_adjustment": lambda X, t, y, **k: regression_adjustment(np.nan_to_num(X), t, y, k.get("estimand", "ATT")),
    "ipw": lambda X, t, y, **k: ipw(np.nan_to_num(X), t, y, k.get("estimand", "ATT")),
    "aipw_parametric": lambda X, t, y, **k: aipw(X, t, y, k.get("estimand", "ATT"), learner="parametric"),
    "aipw_ml": lambda X, t, y, **k: aipw(X, t, y, k.get("estimand", "ATT"), learner="ml", seed=k.get("seed", 0)),
}
