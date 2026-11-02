"""Estimators: unbiasedness where they should be, failure where they should fail, honest SEs."""

import numpy as np
from scipy.special import expit

from adlift.estimators import core
from adlift.results import EffectEstimate


def _obs_data(rng, n=4000, tau=0.5, outcome_nonlinear=False, prop_nonlinear=False):
    x = rng.normal(size=(n, 2))
    lin = 0.8 * x[:, 0] - 0.5 * x[:, 1]
    # exp(.) selection: a linear-logit propensity cannot balance x0 among the most-targeted users
    sel = lin if not prop_nonlinear else np.exp(1.2 * x[:, 0]) - 1.5
    t = (rng.random(n) < expit(sel)).astype(int)
    m0 = x[:, 0] + 0.5 * x[:, 1] + (1.5 * x[:, 0] ** 2 if outcome_nonlinear else 0)
    y = m0 + tau * t + rng.normal(size=n)
    return x, t, y


def test_result_type_basics():
    e = EffectEstimate("ATT", "x", 1.0, 0.5, 100)
    lo, hi = e.ci
    assert lo < 1.0 < hi and e.covers(1.0) and not e.covers(5.0)
    assert 0 < e.pvalue < 1


def test_difference_in_means_unbiased_under_randomization(rng):
    errs = []
    for _ in range(40):
        t = rng.integers(0, 2, 3000)
        y = 1 + 0.3 * t + rng.normal(size=3000)
        errs.append(core.difference_in_means(t, y).estimate - 0.3)
    assert abs(np.mean(errs)) < 0.02


def test_naive_biased_adjusted_unbiased_under_observed_confounding(rng):
    x, t, y = _obs_data(rng, n=20000)
    naive = core.difference_in_means(t, y, "ATT").estimate
    assert naive - 0.5 > 0.3  # targeting inflates the naive lift
    for f in [core.regression_adjustment, core.ipw, core.aipw]:
        e = f(x, t, y, "ATT")
        assert abs(e.estimate - 0.5) < 3.5 * e.se, f.__name__


def test_aipw_double_robustness(rng):
    # outcome model wrong (quadratic missing), propensity right -> IPW and AIPW fine, RA biased
    x, t, y = _obs_data(rng, n=30000, outcome_nonlinear=True)
    ra = core.regression_adjustment(x, t, y, "ATT", outcome_model="ols")
    dr = core.aipw(x, t, y, "ATT")
    assert abs(ra.estimate - 0.5) > 4 * ra.se
    assert abs(dr.estimate - 0.5) < 3.5 * dr.se
    # propensity wrong, outcome right -> RA and AIPW fine, IPW biased
    x, t, y = _obs_data(rng, n=30000, prop_nonlinear=True)
    ip = core.ipw(x, t, y, "ATT")
    dr = core.aipw(x, t, y, "ATT")
    assert abs(ip.estimate - 0.5) > 5 * ip.se
    assert abs(dr.estimate - 0.5) < 3.5 * dr.se


def test_ipw_and_ra_standard_errors_cover(rng):
    hits = {"ipw": [], "ra": []}
    for _ in range(120):
        x, t, y = _obs_data(rng, n=1500)
        hits["ipw"].append(core.ipw(x, t, y, "ATT").covers(0.5))
        hits["ra"].append(core.regression_adjustment(x, t, y, "ATT", outcome_model="ols").covers(0.5))
    for k, v in hits.items():
        assert 0.82 <= np.mean(v) <= 0.98, (k, np.mean(v))


def test_wald_tot_recovers_effect_on_exposed(rng):
    n = 200_000
    z = (rng.random(n) < 0.8).astype(int)
    activity = rng.normal(size=n)
    d = z * (rng.random(n) < expit(-2 + 1.5 * activity))  # only active users get exposed
    p0 = expit(-3 + activity)
    p1 = expit(-3 + activity + 0.8)
    y = (rng.random(n) < np.where(d == 1, p1, p0)).astype(int)
    true_tot = float((p1 - p0)[d == 1].mean())
    tot = core.wald_tot(z, d, y)
    naive = core.difference_in_means(d[z == 1], y[z == 1]).estimate
    assert abs(tot.estimate - true_tot) < 3.5 * tot.se
    assert naive > true_tot * 1.3  # exposed users are more active: naive overstates


def test_missing_outcomes_mar(rng):
    n = 40000
    x = rng.normal(size=(n, 2))
    t = (rng.random(n) < expit(0.7 * x[:, 0])).astype(int)
    y = (rng.random(n) < expit(-0.5 + x[:, 0] + 0.5 * t)).astype(float)
    p_resp = expit(-1 + 0.9 * x[:, 0] + 0.8 * t)  # engaged + exposed answer more
    y_obs = np.where(rng.random(n) < p_resp, y, np.nan)
    truth = float((expit(-0.5 + x[:, 0] + 0.5) - expit(-0.5 + x[:, 0]))[t == 1].mean())
    cc = core.difference_in_means(t, y_obs, "ATT")
    dr = core.aipw(x, t, y_obs, "ATT")
    assert abs(cc.estimate - truth) > 4 * cc.se
    assert abs(dr.estimate - truth) < 3.5 * dr.se


def test_relative_lift_delta_method_matches_bootstrap(rng):
    t = rng.integers(0, 2, 20000)
    y = (rng.random(20000) < np.where(t == 1, 0.06, 0.05)).astype(float)
    e = core.difference_in_means(t, y)
    boots = []
    for _ in range(300):
        i = rng.integers(0, len(t), len(t))
        b = core.difference_in_means(t[i], y[i])
        boots.append(b.relative)
    assert abs(np.std(boots) / e.relative_se - 1) < 0.2


def test_ml_aipw_runs_with_missing_covariates(rng):
    x, t, y = _obs_data(rng, n=6000)
    x[rng.random(len(x)) < 0.1, 1] = np.nan
    e = core.aipw(x, t, y, "ATT", learner="ml", folds=2)
    assert np.isfinite(e.estimate) and e.se > 0


def test_logistic_solver_converges_on_heavy_tailed_rare_outcome(rng):
    """Regression test: a capped Newton step once stalled on Criteo-like data."""
    from scipy.special import expit as sig
    from sklearn.linear_model import LogisticRegression

    n = 100_000
    X = np.column_stack([rng.standard_t(2, n), rng.lognormal(0, 1.5, n), rng.normal(size=n)])
    Xs = (X - X.mean(0)) / X.std(0)
    y = (rng.random(n) < sig(-7 + 1.5 * Xs[:, 0] - 0.8 * Xs[:, 1] + 0.3 * Xs[:, 2])).astype(float)
    g = core.fit_glm("logit", Xs, y)
    p = g.predict()
    assert np.abs(core._add_intercept(Xs).T @ (y - p)).max() / n < 1e-4  # tiny ridge keeps it off exactly 0
    sk = LogisticRegression(penalty=None, max_iter=5000, tol=1e-12).fit(Xs, y)
    assert np.allclose(g.coef, np.r_[sk.intercept_, sk.coef_[0]], atol=5e-2)
