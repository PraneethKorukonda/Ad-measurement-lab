"""Weekly marketing-mix simulator with exact channel contributions and geo experiments.

An advertiser spends across four channels for three years (156 weeks). Revenue
is baseline (trend, seasonality, price, promotions, holidays) plus media
contributions with adstock and Hill saturation, plus an *unobserved demand
shock*. Because the simulator knows every channel's weekly contribution, it
knows the true ROAS, marginal ROAS and the true optimal budget split. That is
what lets us grade the MMM.

Two knobs reproduce the two classic MMM failure modes:

* ``rho`` (channel correlation): when budgets move together, the model cannot
  tell which channel caused the revenue. Estimates trade off against each other
  and intervals widen (or, worse, priors decide the answer).
* ``endogeneity`` (selection bias): search budgets chase demand. When demand
  rises, more people search, search spend rises automatically, and revenue
  rises for reasons search did not cause. A regression on observational spend
  credits search with the demand shock, overstating its ROAS. This is the MMM
  analogue of targeting bias in user-level data.

A **geo holdout experiment** is the fix: switch a channel off in a random 30%
of geographies for 8 weeks. Demand shocks hit test and control geos alike, so
the experiment's lift estimate is unbiased. The MMM can then be *calibrated* by
adding the experiment as an extra likelihood term (see ``adlift.mmm.model``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adlift.mmm.transforms import adstock, hill, lag_matrix

CHANNELS = ["youtube", "search", "social", "display"]
TRUE_PARAMS = {
    #           mean weekly spend, adstock alpha, half-sat K, shape h, target ROAS, endogeneity loading
    "youtube": dict(spend=80_000, alpha=0.60, K=0.9, h=1.4, roas=2.2, endo=0.00),
    "search": dict(spend=120_000, alpha=0.15, K=1.1, h=1.0, roas=3.0, endo=1.00),
    "social": dict(spend=60_000, alpha=0.35, K=0.8, h=1.6, roas=1.4, endo=0.40),
    "display": dict(spend=40_000, alpha=0.25, K=0.6, h=1.2, roas=0.7, endo=0.00),
}
L_MAX = 8


@dataclass
class MMMSimConfig:
    n_weeks: int = 156
    rho: float = 0.2  # correlation of channel budget innovations
    spend_sd: float = 0.35  # log-scale spend variation
    endogeneity: float = 0.0  # log-spend response to the unobserved demand shock
    demand_effect: float = 0.04  # revenue share moved by a 1-SD demand shock
    noise_sd: float = 0.02  # revenue noise as share of baseline
    youtube_flighting: bool = True  # dark periods for YouTube (helps identification)
    # geo experiments: (channel, start week, weeks, spend multiplier in test geos).
    # multiplier 0 = holdout ("go dark"); 2.0 = heavy-up (double spend).
    experiments: tuple = (("search", 110, 8, 0.0), ("search", 60, 10, 2.0))
    experiment_cooldown: int = 6
    n_geos: int = 50
    holdout_share: float = 0.3
    seed: int = 0


@dataclass
class MMMData:
    config: MMMSimConfig
    df: pd.DataFrame  # weekly national data (what the analyst sees)
    spend_history: np.ndarray  # (T + L - 1, C) national spend incl. pre-sample weeks
    truth: dict = field(default_factory=dict)
    experiments: list = field(default_factory=list)
    geo: pd.DataFrame | None = None


def _ar1(rng, T, C, phi, corr):
    cov = np.full((C, C), corr) + (1 - corr) * np.eye(C)
    chol = np.linalg.cholesky(cov)
    e = rng.normal(size=(T, C)) @ chol.T
    z = np.zeros((T, C))
    z[0] = e[0]
    for t in range(1, T):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi**2) * e[t]
    return z


def simulate_mmm(config: MMMSimConfig | None = None, **kw) -> MMMData:
    cfg = config or MMMSimConfig(**kw)
    rng = np.random.default_rng(cfg.seed)
    T, L, C = cfg.n_weeks, L_MAX, len(CHANNELS)
    TT = T + L - 1
    week = np.arange(TT) - (L - 1)  # sample weeks are 0..T-1
    woy = week % 52
    holiday = ((woy >= 46) & (woy <= 51)).astype(float)
    season = np.sin(2 * np.pi * week / 52) + 0.4 * np.cos(4 * np.pi * week / 52)

    demand = np.zeros(TT)
    for t in range(1, TT):
        demand[t] = 0.6 * demand[t - 1] + np.sqrt(1 - 0.36) * rng.normal()

    z = _ar1(rng, TT, C, 0.7, cfg.rho)
    spend = np.zeros((TT, C))
    for j, ch in enumerate(CHANNELS):
        p = TRUE_PARAMS[ch]
        logx = (cfg.spend_sd * z[:, j] + 0.25 * holiday + 0.10 * season
                + cfg.endogeneity * p["endo"] * 0.30 * demand)
        x = np.exp(logx)
        spend[:, j] = p["spend"] * x / x.mean()
    if cfg.youtube_flighting:
        dark = np.zeros(TT, bool)
        for start in rng.choice(np.arange(L, TT - 4, 13), size=4, replace=False):
            dark[start : start + 4] = True
        spend[dark, CHANNELS.index("youtube")] = 0.0

    base_level = 2_000_000.0
    t_s = np.arange(TT) / 52
    price = np.zeros(TT)
    for t in range(1, TT):
        price[t] = 0.8 * price[t - 1] + 0.6 * rng.normal()
    promo = (rng.random(TT) < 0.15).astype(float)
    baseline = base_level * (1 + 0.05 * t_s + 0.08 * season + 0.10 * holiday - 0.03 * price + 0.05 * promo)

    scale = np.array([TRUE_PARAMS[c]["spend"] for c in CHANNELS], float)
    alpha = np.array([TRUE_PARAMS[c]["alpha"] for c in CHANNELS])
    K = np.array([TRUE_PARAMS[c]["K"] for c in CHANNELS])
    h = np.array([TRUE_PARAMS[c]["h"] for c in CHANNELS])

    # beta chosen so that realised ROAS is close to the target ROAS
    lag_base = lag_matrix(spend, L)
    shape = hill(adstock(lag_base, alpha) / scale, K, h)
    beta = np.array([TRUE_PARAMS[c]["roas"] for c in CHANNELS]) * spend[L - 1 :].sum(0) / shape.sum(0)

    # geo experiments: spend multiplied by m in a random share of geos for some weeks.
    # Influence windows (test weeks + carryover) must not overlap across experiments.
    contrib_base = beta * hill(adstock(lag_base, alpha) / scale, K, h)
    contrib = contrib_base.copy()
    national_spend = spend.copy()
    exp_specs = []
    for (ch, s0, w, mult) in cfg.experiments:
        shares = rng.dirichlet(np.full(cfg.n_geos, 3.0)) if not exp_specs else exp_specs[0]["shares"]
        treat = np.zeros(cfg.n_geos, bool)
        treat[rng.permutation(cfg.n_geos)[: int(round(cfg.holdout_share * cfg.n_geos))]] = True
        f = float(shares[treat].sum())
        jx = CHANNELS.index(ch)
        test_spend = spend.copy()
        rows = slice(s0 + L - 1, s0 + L - 1 + w)
        test_spend[rows, jx] *= mult
        c_test = beta * hill(adstock(lag_matrix(test_spend, L), alpha) / scale, K, h)
        delta = c_test - contrib_base  # per-capita change in test geos, (T, C)
        contrib = contrib + f * delta
        national_spend = national_spend + f * (test_spend - spend)
        exp_specs.append(dict(channel=ch, start=s0, weeks=w, multiplier=mult, shares=shares, treat=treat,
                              f=f, jx=jx, delta=delta, extra_spend=(test_spend - spend)[L - 1 :, jx]))

    eps = cfg.noise_sd * base_level * rng.normal(size=TT)
    dem = cfg.demand_effect * base_level * demand
    sample = slice(L - 1, None)
    revenue = baseline[sample] + contrib.sum(1) + dem[sample] + eps[sample]

    df = pd.DataFrame({"week": np.arange(T),
                       "date": pd.Timestamp("2023-01-02") + pd.to_timedelta(7 * np.arange(T), "D"),
                       "revenue": revenue})
    for j, ch in enumerate(CHANNELS):
        df[f"spend_{ch}"] = national_spend[sample, j]
    df["price_index"] = price[sample]
    df["promo"] = promo[sample]
    df["holiday"] = holiday[sample]
    # Organic search query volume: a noisy proxy for the demand shock (observable in
    # practice, e.g. from Google Trends). Chen et al. (2018) use it to de-bias search.
    df["query_volume"] = demand[sample] + 0.3 * rng.normal(size=T)

    truth = {
        "params": {"alpha": alpha, "K": K, "h": h, "beta": beta, "scale": scale},
        "contribution": contrib,  # (T, C) sample weeks only
        "roas": dict(zip(CHANNELS, contrib.sum(0) / national_spend[sample].sum(0))),
        "media_share": float(contrib.sum() / revenue.sum()),
        "demand": demand[sample],
    }
    truth["mroas"] = dict(zip(CHANNELS, marginal_roas(national_spend, alpha, K, h, beta, scale)))

    geo, exps = None, []
    if exp_specs:
        G = cfg.n_geos
        shares = exp_specs[0]["shares"]
        geo_noise = rng.normal(size=(T, G)) * cfg.noise_sd * base_level * np.sqrt(shares)[None, :]
        pc = (baseline[sample] + dem[sample] + contrib_base.sum(1))[:, None] * np.ones((1, G))
        for e in exp_specs:
            pc = pc + np.where(e["treat"][None, :], e["delta"].sum(1)[:, None], 0.0)
        geo_rev = shares[None, :] * pc + geo_noise
        geo = pd.DataFrame({"week": np.repeat(np.arange(T), G), "geo": np.tile(np.arange(G), T),
                            "share": np.tile(shares, T), "revenue": geo_rev.ravel()})
        for k, e in enumerate(exp_specs):
            geo[f"test_{k}"] = np.tile(e["treat"], T).astype(int)
            s0, w = e["start"], e["weeks"]
            post_end = s0 + w + cfg.experiment_cooldown
            true_lift = float(e["f"] * e["delta"][s0:post_end, e["jx"]].sum())
            spend_change = float(e["f"] * e["extra_spend"][s0 : s0 + w].sum())
            est, se = geo_experiment_lift(geo, f"test_{k}", pre=(s0 - 26, s0), post=(s0, post_end))
            exps.append({"channel": e["channel"], "start": s0, "weeks": w, "cooldown": cfg.experiment_cooldown,
                         "multiplier": e["multiplier"], "test_share": e["f"],
                         "lift_estimate": est, "lift_se": se, "true_lift": true_lift,
                         "spend_change": spend_change, "iroas_estimate": est / spend_change,
                         "iroas_se": abs(se / spend_change), "true_iroas": true_lift / spend_change})
    return MMMData(config=cfg, df=df, spend_history=national_spend, truth=truth, experiments=exps, geo=geo)


def geo_experiment_lift(geo: pd.DataFrame, test_col: str, pre: tuple[int, int],
                        post: tuple[int, int]) -> tuple[float, float]:
    """Revenue caused by the test treatment in test geos, by weighted ANCOVA on per-capita revenue.

    For each geo: y = per-capita revenue summed over the post window, x = the
    same over a pre window. Regress y on [1, x, holdout] with weights = geo
    population share (larger geos are less noisy). The holdout coefficient is
    the per-capita revenue change; times the holdout population share it is the
    total incremental revenue the channel generated there (sign flipped). This
    is a simplified version of Google's geo-based regression (Vaver & Koehler 2011).
    """
    g = geo.groupby("geo").agg(share=("share", "first"), holdout=(test_col, "first"))
    pre_rev = geo[(geo.week >= pre[0]) & (geo.week < pre[1])].groupby("geo")["revenue"].sum()
    post_rev = geo[(geo.week >= post[0]) & (geo.week < post[1])].groupby("geo")["revenue"].sum()
    s = g["share"].to_numpy()
    y = post_rev.to_numpy() / s
    x = pre_rev.to_numpy() / s
    d = g["holdout"].to_numpy()
    X = np.column_stack([np.ones_like(y), x, d])
    W = s
    XtW = X.T * W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    resid = y - X @ beta
    # HC1 sandwich
    bread = np.linalg.inv(XtW @ X)
    meat = (X.T * (W**2 * resid**2)) @ X
    cov = bread @ meat @ bread * len(y) / (len(y) - X.shape[1])
    f = s[d == 1].sum()
    return float(beta[2] * f), float(np.sqrt(cov[2, 2]) * f)


def marginal_roas(spend_hist, alpha, K, h, beta, scale, bump: float = 0.01) -> np.ndarray:
    """Revenue from the next dollar: d revenue / d spend at current levels (all weeks +1%)."""
    L = L_MAX
    base = beta * hill(adstock(lag_matrix(spend_hist, L), alpha) / scale, K, h)
    out = []
    for j in range(spend_hist.shape[1]):
        s2 = spend_hist.copy()
        s2[:, j] *= 1 + bump
        up = beta * hill(adstock(lag_matrix(s2, L), alpha) / scale, K, h)
        out.append((up[:, j].sum() - base[:, j].sum()) / (bump * spend_hist[L - 1 :, j].sum()))
    return np.array(out)


SCENARIOS = {
    "clean": dict(rho=0.2, endogeneity=0.0),
    "correlated": dict(rho=0.9, endogeneity=0.0),
    "endogenous": dict(rho=0.2, endogeneity=1.0),
}
