"""MMM transforms, simulator truth, geo experiments, optimizer and (slow) model recovery."""

import numpy as np
import pytest

from adlift.mmm.optimize import optimize_allocation
from adlift.mmm.transforms import adstock, adstock_weights, hill, lag_matrix
from adlift.sim.mmm import CHANNELS, MMMSimConfig, simulate_mmm


def test_lag_matrix_alignment():
    x = np.arange(10, dtype=float)
    lm = lag_matrix(x, 3)
    assert lm.shape == (8, 3, 1)
    assert lm[0, :, 0].tolist() == [2.0, 1.0, 0.0]


def test_normalised_adstock_steady_state_and_hill():
    w = adstock_weights(np.array([0.5, 0.9]), 8)
    assert np.allclose(w.sum(0), 1)
    x = np.full((20, 2), 3.0)
    assert np.allclose(adstock(lag_matrix(x, 8), np.array([0.5, 0.9])), 3.0)
    assert np.isclose(hill(np.array(2.0), 2.0, 1.7), 0.5)
    s = np.linspace(0, 5, 50)
    assert np.all(np.diff(hill(s, 1.0, 1.3)) >= 0)


def test_simulator_roas_and_marginal_roas():
    d = simulate_mmm(MMMSimConfig(seed=1))
    for c, target in zip(CHANNELS, [2.2, 3.0, 1.4, 0.7]):
        assert abs(d.truth["roas"][c] - target) / target < 0.05
        assert d.truth["mroas"][c] < d.truth["roas"][c]  # diminishing returns
    assert 0.1 < d.truth["media_share"] < 0.35


def test_correlation_and_endogeneity_knobs():
    hi = simulate_mmm(MMMSimConfig(seed=2, rho=0.9))
    lo = simulate_mmm(MMMSimConfig(seed=2, rho=0.0))
    corr = lambda d: np.corrcoef(d.df.spend_search, d.df.spend_social)[0, 1]  # noqa: E731
    assert corr(hi) > corr(lo) + 0.3
    en = simulate_mmm(MMMSimConfig(seed=2, endogeneity=1.0))
    assert np.corrcoef(np.log(en.df.spend_search), en.truth["demand"])[0, 1] > 0.3


def test_geo_experiments_are_unbiased_even_with_endogenous_spend():
    z = {0: [], 1: []}
    for s in range(40):
        d = simulate_mmm(MMMSimConfig(seed=s, endogeneity=1.0))
        for k, e in enumerate(d.experiments):
            z[k].append((e["lift_estimate"] - e["true_lift"]) / e["lift_se"])
    for k in z:
        assert abs(np.mean(z[k])) < 0.4 and 0.7 < np.std(z[k]) < 1.4


def test_optimizer_respects_budget_and_bounds_and_improves():
    beta = np.array([[3.0, 1.0, 2.0]])
    K = np.array([[1.0, 1.0, 1.0]])
    h = np.array([[1.0, 1.0, 1.0]])
    scale = np.ones(3)
    cur = np.array([1.0, 1.0, 1.0])
    rec = optimize_allocation(beta, K, h, scale, 3.0, cur, band=(0.5, 1.5))
    assert np.isclose(rec.sum(), 3.0)
    assert np.all(rec >= 0.5 - 1e-9) and np.all(rec <= 1.5 + 1e-9)
    tot = lambda s: float(np.sum(beta[0] * s / (s + 1)))  # noqa: E731
    assert tot(rec) > tot(cur)
    assert rec[0] > rec[1]


@pytest.mark.slow
def test_mmm_recovers_marginal_roas_on_clean_data():
    from adlift.mmm.model import fit_mmm, prepare

    d = simulate_mmm(MMMSimConfig(seed=11))
    inp = prepare(d.df, CHANNELS, ["price_index", "promo", "holiday"], spend_history=d.spend_history)
    fit = fit_mmm(inp, draws=400, tune=500, seed=1)
    m = fit.mroas()
    covered = [np.quantile(m[:, j], 0.02) <= d.truth["mroas"][c] <= np.quantile(m[:, j], 0.98)
               for j, c in enumerate(CHANNELS)]
    assert sum(covered) >= 3
    assert fit.diagnostics["max_rhat"] < 1.05
