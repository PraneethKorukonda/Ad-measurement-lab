"""Viewer-experience guardrail experiment for monetization policies.

Question: does a more profitable ad policy hurt viewers, and would a standard
one- or two-week A/B test notice?

Model. Each viewer has a session rate, a typical session length and a segment
(country tier x favourite content). Sessions have an ad break every 8 minutes
(pre-roll at minute 0). Each break shows 1 ad, or 2 under a "double pod"
policy, drawn from that policy's auction outcomes in the viewer's segment
(filled or not, price, creative quality). At each break the viewer may abandon
the session; abandonment rises with pod size and falls with ad quality.

Viewers also *learn*: each ad seen lowers an "ad tolerance" state that decays
back slowly (85% persistence per week), and lower tolerance reduces how often
they come back. This is the mechanism documented by Hohnhold, O'Brien and Tang
(2015, "Focusing on the Long-term: It's Good for Users and Business"): the
revenue gain from heavier ad load shows up immediately, while the engagement
loss accumulates over weeks, so short tests overstate the net gain.

Design. 60K viewers, 12 burn-in weeks under the status quo, 2 pre-period weeks
(used for CUPED-style regression adjustment), 6 treatment weeks with viewers
randomized to three arms, and 2 post-period weeks where everyone returns to the
status quo (the persistent effect then isolates learning). Every viewer is
simulated under every arm with common random numbers, so the true effect per
week is known exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from adlift.auction import reserves as R
from adlift.auction.market import draw_day, make_market, simulate_day
from adlift.estimators.core import regression_adjustment


@dataclass
class AdPolicy:
    name: str
    pod_size: int
    # per segment (15): arrays of slot outcomes to sample from
    filled: list
    price: list
    quality: list
    filled2: list | None = None
    price2: list | None = None
    quality2: list | None = None


def _slot_pools(res, seg):
    f, p, q = [], [], []
    for s in range(15):
        m = seg == s
        f.append(res.filled[m])
        p.append(res.price[m])
        q.append(np.nan_to_num(res.winner_quality[m]))
    return f, p, q


def build_policies(seed: int = 0, n_opportunities: int = 150_000) -> dict[str, AdPolicy]:
    """Auction outcome pools for: status quo, optimized floors, and double pods at the status-quo floor.

    Arm C changes only ad load (one change per arm), so its effect is attributable."""
    market = make_market(seed=seed)
    train = draw_day(market, n_opportunities, day=0)
    logged = simulate_day(market, train, R.fixed(2.0))
    floors = R.myerson(logged.bids, train.seg)
    day = draw_day(market, n_opportunities, day=50)
    second = draw_day(market, n_opportunities, day=51, value_scale=0.6)  # 2nd ad in a pod: less attention
    sq = simulate_day(market, day, R.fixed(2.0))
    opt = simulate_day(market, day, floors)
    sq2 = simulate_day(market, second, R.fixed(2.0))
    out = {
        "A_status_quo": AdPolicy("A_status_quo", 1, *_slot_pools(sq, day.seg)),
        "B_optimized_floors": AdPolicy("B_optimized_floors", 1, *_slot_pools(opt, day.seg)),
    }
    f, p, q = _slot_pools(sq, day.seg)
    f2, p2, q2 = _slot_pools(sq2, second.seg)
    out["C_double_pods"] = AdPolicy("C_double_pods", 2, f, p, q, f2, p2, q2)
    return out


@dataclass
class ViewerConfig:
    n_viewers: int = 60_000
    burn_in: int = 12
    pre_weeks: int = 2
    treat_weeks: int = 6
    post_weeks: int = 2
    break_every_min: float = 8.0
    tolerance_decay: float = 0.85
    return_sensitivity: float = 0.30  # how much low tolerance suppresses sessions
    annoyance_per_ad: float = 0.0025
    annoyance_second_slot: float = 0.001
    annoyance_quality: float = 0.001
    second_ad_abandon_logit: float = 0.45
    seed: int = 0


def _viewers(cfg: ViewerConfig):
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_viewers
    tier = rng.choice(3, n, p=[0.35, 0.35, 0.30])
    content = rng.choice(5, n, p=[0.15, 0.2, 0.3, 0.25, 0.1])
    rate = rng.gamma(2.0, 2.0, n)  # sessions / week at full tolerance
    minutes = np.exp(rng.normal(np.log(18), 0.5, n))
    return pd.DataFrame({"viewer_id": np.arange(n), "segment": tier * 5 + content, "tier": tier,
                         "session_rate": rate, "typical_minutes": minutes})


def _simulate_week(v: pd.DataFrame, tol: np.ndarray, pol: AdPolicy, cfg: ViewerConfig,
                   rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    n = len(v)
    lam = v["session_rate"].to_numpy() * np.exp(cfg.return_sensitivity * tol)
    n_sess = rng.poisson(lam)
    owner = np.repeat(np.arange(n), n_sess)
    S = len(owner)
    length = v["typical_minutes"].to_numpy()[owner] * np.exp(rng.normal(0, 0.4, S))
    n_breaks = 1 + np.floor(length / cfg.break_every_min).astype(int)
    b_sess = np.repeat(np.arange(S), n_breaks)
    b_pos = np.arange(len(b_sess)) - np.repeat(np.cumsum(n_breaks) - n_breaks, n_breaks)
    b_user = owner[b_sess]
    seg = v["segment"].to_numpy()[b_user]
    B = len(b_sess)

    filled = np.zeros((B, pol.pod_size), bool)
    price = np.zeros((B, pol.pod_size))
    qual = np.zeros((B, pol.pod_size))
    u_pick = rng.random((B, 2))
    for s in range(15):
        m = np.flatnonzero(seg == s)
        if len(m) == 0:
            continue
        pools = [(pol.filled, pol.price, pol.quality)]
        if pol.pod_size == 2:
            pools.append((pol.filled2, pol.price2, pol.quality2))
        for k, (F, P, Q) in enumerate(pools):
            idx = (u_pick[m, k] * len(F[s])).astype(int)
            filled[m, k] = F[s][idx]
            price[m, k] = P[s][idx]
            qual[m, k] = Q[s][idx]
    n_ads = filled.sum(axis=1)
    mean_q = np.where(n_ads > 0, (qual * filled).sum(axis=1) / np.maximum(n_ads, 1), 0.0)
    p_abandon = np.where(n_ads > 0, expit(-3.6 + cfg.second_ad_abandon_logit * (n_ads - 1) - 0.35 * mean_q - 0.8 * tol[b_user]), 0.004)
    aband = rng.random(B) < p_abandon
    # first abandoning break per session (breaks after it never happen)
    first_ab = np.full(S, np.iinfo(np.int64).max)
    np.minimum.at(first_ab, b_sess[aband], b_pos[aband])
    shown = b_pos <= first_ab[b_sess]
    watched = np.where(first_ab < np.iinfo(np.int64).max,
                       np.minimum(length, first_ab * cfg.break_every_min), length)

    ads_seen = np.bincount(b_user[shown], weights=n_ads[shown], minlength=n)
    revenue = np.bincount(b_user[shown], weights=(price * filled).sum(axis=1)[shown], minlength=n)
    annoy_b = (filled[:, 0] * (cfg.annoyance_per_ad - cfg.annoyance_quality * qual[:, 0]))
    if pol.pod_size == 2:
        annoy_b = annoy_b + filled[:, 1] * (cfg.annoyance_per_ad + cfg.annoyance_second_slot
                                            - cfg.annoyance_quality * qual[:, 1])
    annoy = np.bincount(b_user[shown], weights=np.maximum(annoy_b, 0)[shown], minlength=n)
    new_tol = cfg.tolerance_decay * tol - annoy
    out = {
        "sessions": np.bincount(owner, minlength=n).astype(float),
        "watch_minutes": np.bincount(owner, weights=watched, minlength=n),
        "ads_seen": ads_seen,
        "revenue": revenue,
        "breaks_reached": np.bincount(b_user[shown], minlength=n).astype(float),
        "abandons": np.bincount(b_user[aband & shown], minlength=n).astype(float),
    }
    return out, new_tol


def run_viewer_experiment(cfg: ViewerConfig | None = None, policies: dict[str, AdPolicy] | None = None,
                          coverage_draws: int = 100) -> dict[str, pd.DataFrame]:
    cfg = cfg or ViewerConfig()
    pols = policies or build_policies(cfg.seed)
    v = _viewers(cfg)
    n = len(v)
    tol = np.zeros(n)
    control = pols["A_status_quo"]
    for w in range(cfg.burn_in):
        _, tol = _simulate_week(v, tol, control, cfg, np.random.default_rng(10_000 + w))

    pre = []
    for w in range(cfg.pre_weeks):
        out, tol = _simulate_week(v, tol, control, cfg, np.random.default_rng(20_000 + w))
        pre.append(out)
    pre_tot = {k: sum(p[k] for p in pre) for k in pre[0]}

    # potential outcomes under each arm (common random numbers by week)
    weeks = cfg.treat_weeks + cfg.post_weeks
    po = {}
    for name, pol in pols.items():
        t = tol.copy()
        rows = []
        for w in range(weeks):
            use = pol if w < cfg.treat_weeks else control
            out, t = _simulate_week(v, t, use, cfg, np.random.default_rng(30_000 + w))
            rows.append(out)
        po[name] = rows

    rng = np.random.default_rng(cfg.seed + 99)
    arms = list(pols)
    assign = rng.integers(0, len(arms), n)
    results = []
    for w in range(weeks):
        for k, arm in enumerate(arms[1:], start=1):
            m = (assign == 0) | (assign == k)
            t = (assign[m] == k).astype(int)
            for metric in ["revenue", "watch_minutes", "sessions", "ads_seen"]:
                y = np.where(assign == k, po[arm][w][metric], po[arms[0]][w][metric])[m]
                x = pre_tot[metric][m][:, None]
                est = regression_adjustment(x, t, y, "ATE", outcome_model="ols")
                truth = float(np.mean(po[arm][w][metric] - po[arms[0]][w][metric]))
                base = float(np.mean(po[arms[0]][w][metric]))
                results.append({
                    "week": w + 1, "phase": "treatment" if w < cfg.treat_weeks else "post",
                    "arm": arm, "metric": metric, "estimate": est.estimate, "se": est.se,
                    "ci_low": est.ci[0], "ci_high": est.ci[1], "relative_pct": 100 * est.estimate / base,
                    "true_effect": truth, "true_relative_pct": 100 * truth / base, "covers": est.covers(truth),
                    "control_mean": base,
                })
    weekly = pd.DataFrame(results)

    # abandonment rate per arm over treatment weeks (descriptive)
    ab = []
    for arm in arms:
        reached = sum(po[arm][w]["breaks_reached"].sum() for w in range(cfg.treat_weeks))
        abn = sum(po[arm][w]["abandons"].sum() for w in range(cfg.treat_weeks))
        ab.append({"arm": arm, "ad_break_abandon_rate": abn / reached})

    # cumulative effects over treatment window, and naive 1-week vs. 6-week readout
    cum = []
    for arm in arms[1:]:
        for metric in ["revenue", "watch_minutes", "sessions"]:
            g = weekly[(weekly.arm == arm) & (weekly.metric == metric)]
            cum.append({"arm": arm, "metric": metric,
                        "week1_relative_pct": g[g.week == 1]["true_relative_pct"].item(),
                        "week6_relative_pct": g[g.week == cfg.treat_weeks]["true_relative_pct"].item(),
                        "post_period_relative_pct": g[g.phase == "post"]["true_relative_pct"].mean(),
                        "week1_estimate_pct": g[g.week == 1]["relative_pct"].item(),
                        "week6_estimate_pct": g[g.week == cfg.treat_weeks]["relative_pct"].item()})
    # Randomization-based check of interval coverage. The weekly cells above all come
    # from ONE random split of the same viewers, so they are strongly correlated: one
    # unlucky split shifts every week together. Re-drawing the split many times (the
    # potential outcomes are already simulated) measures coverage properly.
    cov_rows = []
    for draw in range(coverage_draws):
        a2 = np.random.default_rng(10_000 + draw).integers(0, len(arms), n)
        for w in [0, cfg.treat_weeks - 1]:
            for k, arm in enumerate(arms[1:], start=1):
                m = (a2 == 0) | (a2 == k)
                t = (a2[m] == k).astype(int)
                for metric in ["revenue", "watch_minutes"]:
                    y = np.where(a2 == k, po[arm][w][metric], po[arms[0]][w][metric])[m]
                    est = regression_adjustment(pre_tot[metric][m][:, None], t, y, "ATE", outcome_model="ols")
                    truth = float(np.mean(po[arm][w][metric] - po[arms[0]][w][metric]))
                    cov_rows.append({"arm": arm, "metric": metric, "week": w + 1, "covered": est.covers(truth),
                                     "z": (est.estimate - truth) / est.se})
    cov = pd.DataFrame(cov_rows)
    coverage = cov.groupby(["arm", "metric", "week"]).agg(coverage=("covered", "mean"), z_sd=("z", "std"),
                                                          draws=("covered", "size")).reset_index()

    return {"weekly": weekly, "abandonment": pd.DataFrame(ab), "summary": pd.DataFrame(cum),
            "coverage_check": coverage,
            "assignment_counts": pd.Series(assign).map(dict(enumerate(arms))).value_counts().rename_axis("arm").reset_index()}
