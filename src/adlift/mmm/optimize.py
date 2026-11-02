"""Budget allocation from posterior response curves.

The optimizer chooses a steady weekly spend per channel that maximises the
posterior-mean total media contribution, holding total budget fixed and keeping
each channel within a band around current spend (large moves extrapolate the
curves beyond the data that identified them; bounds are a guardrail, not a
nicety). Under constant weekly spend, normalised adstock equals spend, so
weekly contribution is beta * Hill(spend / scale).

Because we keep posterior draws, the output is a *distribution* of uplift:
we report the expected gain, its interval and P(gain > 0). Against the
simulator we can also score the recommendation with the true curves: what
share of the truly achievable gain did it capture?
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from adlift.mmm.transforms import hill


def _total(spend, beta, K, h, scale):
    return np.sum(beta * hill(spend / scale, K, h), axis=-1)


def optimize_allocation(beta, K, h, scale, budget: float, current: np.ndarray,
                        band: tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
    """Maximise mean over draws of total weekly contribution. beta, K, h: (draws, C)."""
    beta, K, h = np.atleast_2d(beta), np.atleast_2d(K), np.atleast_2d(h)
    current = np.asarray(current, float)
    bounds = [(band[0] * c, band[1] * c) for c in current]

    def neg(x):
        return -float(np.mean(_total(x[None, :], beta, K, h, scale)))

    x0 = current * budget / current.sum()
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    res = optimize.minimize(neg, x0, method="SLSQP", bounds=bounds,
                            constraints=[{"type": "eq", "fun": lambda x: x.sum() - budget}],
                            options={"maxiter": 500, "ftol": 1e-10})
    return res.x


def recommend(fit, band: tuple[float, float] = (0.5, 1.5), max_draws: int = 400, seed: int = 0) -> dict:
    """Optimal reallocation of the current average weekly budget, with uncertainty."""
    rng = np.random.default_rng(seed)
    n = len(fit.draws["beta"])
    idx = rng.choice(n, size=min(max_draws, n), replace=False)
    beta = fit.draws["beta"][idx] * fit.inputs.y_scale
    K, h = fit.draws["K"][idx], fit.draws["h"][idx]
    scale = fit.inputs.spend_scale
    current = fit.inputs.spend.mean(axis=0)
    budget = current.sum()
    rec = optimize_allocation(beta, K, h, scale, budget, current, band)
    gain = _total(rec[None, :], beta, K, h, scale) - _total(current[None, :], beta, K, h, scale)
    return {"channels": fit.channels, "current": current, "recommended": rec, "budget": budget,
            "gain_draws": gain, "gain_mean": float(gain.mean()),
            "gain_ci": tuple(np.quantile(gain, [0.05, 0.95])), "p_gain_positive": float((gain > 0).mean())}


def score_against_truth(rec: dict, true_params: dict, band: tuple[float, float] = (0.5, 1.5)) -> dict:
    """Evaluate a recommendation with the true response curves."""
    b, K, h, scale = (true_params[k] for k in ["beta", "K", "h", "scale"])
    cur, r = rec["current"], rec["recommended"]
    best = optimize_allocation(b, K, h, scale, rec["budget"], cur, band)
    f = lambda x: float(_total(x[None, :], b[None, :], K[None, :], h[None, :], scale)[0])  # noqa: E731
    achievable = f(best) - f(cur)
    realised = f(r) - f(cur)
    return {"true_gain_of_recommendation": realised, "true_achievable_gain": achievable,
            "share_of_achievable_gain": realised / achievable if achievable > 0 else np.nan,
            "true_optimal": best, "predicted_gain": rec["gain_mean"]}
