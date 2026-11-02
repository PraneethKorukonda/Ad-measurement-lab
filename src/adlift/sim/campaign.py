"""User-level ad campaign simulator with exact ground truth.

Each simulated user has observable attributes, an unobserved purchase intent,
an eligible campaign, an exposure decision, an impression count, and *both*
potential outcomes for three metrics:

* ``converted``  - purchase in the campaign window (observed for everyone)
* ``aware``      - aided brand awareness (brand-lift survey; observed only for respondents)
* ``intent``     - purchase intent (same survey)

Because the simulator knows P(Y=1 | exposed) and P(Y=1 | not exposed) for every
user, the true ATT/ATE, per-segment effects, per-campaign effects and the true
frequency dose-response are all exact. That is what makes the estimators
*benchmarkable*: on real data we never see both potential outcomes.

Scenarios switch on one failure mode at a time (see ``SCENARIOS``):

randomized              exposure is a coin flip (holdout test): every estimator should work
targeted                exposure depends on observed covariates, linearly in the logit
outcome_misspec         outcomes are nonlinear in covariates -> linear outcome models are wrong
propensity_misspec      targeting is nonlinear -> a linear-logit propensity model is wrong
both_misspec            both wrong -> only flexible (ML) nuisance models can recover
weak_overlap            aggressive targeting -> propensities near 0/1, IPW variance explodes
unobserved_confounding  latent intent drives both targeting and outcomes -> nothing observational works
survey_nonresponse      exposed and engaged users answer the survey more -> complete-case lift is biased
null_effect             ads do nothing -> false positive rate check
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.special import expit, logit

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
REGIONS = ["midwest", "northeast", "south", "west"]
FORMATS = ["display", "native", "video"]
CPM = {"display": 6.0, "native": 10.0, "video": 25.0}


@dataclass(frozen=True)
class CampaignConfig:
    n_users: int = 40_000
    n_campaigns: int = 16
    selection: str = "linear"  # random | linear | nonlinear
    selection_strength: float = 1.0
    target_exposure_rate: float = 0.40
    outcome_nonlinear: bool = False
    latent_confounding: float = 0.0
    effect_scale: float = 1.0
    nonresponse: str = "mcar"  # mcar | mar
    survey_rate: float = 0.30
    seed: int = 0


SCENARIOS: dict[str, dict] = {
    "randomized": dict(selection="random"),
    "targeted": dict(selection="linear"),
    "outcome_misspec": dict(selection="linear", outcome_nonlinear=True),
    "propensity_misspec": dict(selection="nonlinear"),
    "both_misspec": dict(selection="nonlinear", outcome_nonlinear=True),
    "weak_overlap": dict(selection="linear", selection_strength=2.6),
    "unobserved_confounding": dict(selection="linear", latent_confounding=1.0),
    "survey_nonresponse": dict(selection="linear", nonresponse="mar"),
    "null_effect": dict(selection="linear", effect_scale=0.0),
}


def scenario_config(name: str, **overrides) -> CampaignConfig:
    return replace(CampaignConfig(**SCENARIOS[name]), **overrides)


# Covariates an analyst would use (all observed, with missing income).
NUMERIC = ["log_sessions", "affinity", "income"]
CATEGORICAL = ["age_group", "region", "device", "prior_purchaser", "campaign_id"]


@dataclass
class CampaignData:
    config: CampaignConfig
    users: pd.DataFrame  # one row per user: observables, exposure, outcomes, truth
    campaigns: pd.DataFrame
    truth: dict = field(default_factory=dict)

    @property
    def observed(self) -> pd.DataFrame:
        return self.users[[c for c in self.users.columns if not c.startswith("true_") and c != "latent_intent"]]


def _frequency_multiplier(f: np.ndarray) -> np.ndarray:
    """Concave dose-response, normalised to 1 at 3 impressions."""
    return (1 - np.exp(-0.5 * f)) / (1 - np.exp(-1.5))


def simulate_campaign(config: CampaignConfig | None = None, **kw) -> CampaignData:
    cfg = config or CampaignConfig(**kw)
    rng = np.random.default_rng(cfg.seed)
    n, K = cfg.n_users, cfg.n_campaigns

    # --- campaigns -----------------------------------------------------------
    crng = np.random.default_rng(10_000 + cfg.seed)
    fmt = np.array([FORMATS[i % 3] for i in range(K)])
    crng.shuffle(fmt)
    camp = pd.DataFrame({
        "campaign_id": [f"C{i:02d}" for i in range(K)],
        "advertiser": [f"adv_{i // 4}" for i in range(K)],
        "format": fmt,
        "objective": np.where(crng.random(K) < 0.5, "awareness", "consideration"),
        "frequency_cap": crng.choice([3, 6, 10], size=K),
        "creative_score": np.round(crng.normal(0, 1, K), 3),  # pre-launch creative test score
        "start_date": pd.Timestamp("2026-03-02") + pd.to_timedelta(crng.integers(0, 28, K), "D"),
    })
    camp["end_date"] = camp["start_date"] + pd.Timedelta(days=27)
    fmt_conv = {"display": 0.0, "native": 0.10, "video": 0.05}
    fmt_aware = {"display": 0.0, "native": 0.10, "video": 0.35}

    # --- users ---------------------------------------------------------------
    age = rng.choice(len(AGE_GROUPS), size=n, p=[0.15, 0.27, 0.23, 0.18, 0.17])
    region = rng.choice(len(REGIONS), size=n)
    mobile = rng.random(n) < 0.65
    act = rng.normal(0, 1, n)
    sessions = rng.poisson(np.exp(1.6 + 0.7 * act))
    log_sessions = np.log1p(sessions)
    eng = (log_sessions - log_sessions.mean()) / log_sessions.std()
    aff_z = rng.normal(0, 1, n)
    affinity = expit(aff_z)
    income_z = rng.normal(0, 1, n) + 0.3 * (age >= 2)
    prior = rng.random(n) < expit(-1.6 + 0.25 * age + 0.4 * eng)
    U = rng.normal(0, 1, n)  # unobserved purchase intent
    cidx = rng.integers(0, K, n)
    young = age <= 1
    old = age == 4

    # Observed income has missing values, more often on mobile (MAR, not MCAR).
    income_obs = np.where(rng.random(n) < np.where(mobile, 0.20, 0.08), np.nan, np.round(income_z, 3))

    # --- exposure ------------------------------------------------------------
    camp_icpt = crng.normal(0, 0.3, K)[cidx]
    if cfg.selection == "random":
        score = np.zeros(n)
    elif cfg.selection == "linear":
        score = 0.8 * eng + 0.6 * prior + 0.6 * aff_z + 0.3 * mobile - 0.3 * old + camp_icpt
    elif cfg.selection == "nonlinear":
        score = (1.1 * (np.abs(eng) > 0.9) + 0.9 * aff_z * eng + 0.7 * prior * (aff_z > 0)
                 - 0.6 * (eng < -1) + camp_icpt)
    else:  # pragma: no cover
        raise ValueError(cfg.selection)
    score = cfg.selection_strength * score + cfg.latent_confounding * 1.0 * U
    a0 = _calibrate_intercept(score, cfg.target_exposure_rate)
    e_true = expit(a0 + score) if cfg.selection != "random" or cfg.latent_confounding else np.full(n, cfg.target_exposure_rate)
    exposed = rng.random(n) < e_true

    # Would-be frequency (realised only if exposed): engaged users see more ads.
    cap = camp["frequency_cap"].to_numpy()[cidx]
    freq_pot = np.minimum(cap, 1 + rng.poisson(np.exp(0.4 + 0.5 * eng)))
    frequency = np.where(exposed, freq_pot, 0)
    fm = _frequency_multiplier(freq_pot)

    # --- outcomes (logit scale) ---------------------------------------------
    nl = cfg.outcome_nonlinear
    gU = cfg.latent_confounding
    age_c = np.array([0.0, 0.1, 0.15, 0.1, -0.05])[age]
    cb = crng.normal(0, 0.15, K)[cidx]
    creative = camp["creative_score"].to_numpy()[cidx]
    camp_fmt = camp["format"].to_numpy()[cidx]
    camp_obj = camp["objective"].to_numpy()[cidx]
    s = cfg.effect_scale

    conv0 = (-3.0 + 0.5 * eng + 0.9 * prior + 0.4 * aff_z + 0.15 * income_z + age_c + cb + 0.8 * gU * U
             + (nl * (0.7 * eng**2 - 0.9 * (aff_z > 1) + 0.6 * np.sin(2 * aff_z))))
    tau_conv = s * fm * (0.30 + 0.25 * young - 0.25 * prior + np.vectorize(fmt_conv.get)(camp_fmt)
                         + 0.08 * creative)
    aware0 = (-0.6 + 0.25 * eng + 0.3 * aff_z + 0.3 * young + 0.5 * gU * U + cb
              + nl * (0.6 * eng**2 - 0.8 * np.abs(aff_z)))
    tau_aware = s * fm * (0.25 + np.vectorize(fmt_aware.get)(camp_fmt) + 0.10 * (camp_obj == "awareness")
                          + 0.15 * young - 0.10 * old + 0.10 * creative)
    intent0 = (-1.8 + 0.3 * eng + 0.5 * aff_z + 0.4 * prior + 0.1 * income_z + 0.6 * gU * U + cb
               + nl * (0.5 * eng**2 - 0.7 * (aff_z > 1)))
    tau_intent = s * fm * (0.15 + 0.10 * (camp_obj == "consideration") + 0.05 * young + 0.06 * creative)

    out = {}
    for name, l0, tau in [("converted", conv0, tau_conv), ("aware", aware0, tau_aware), ("intent", intent0, tau_intent)]:
        p0, p1 = expit(l0), expit(l0 + tau)
        u = rng.random(n)
        y0, y1 = (u < p0).astype(int), (u < p1).astype(int)
        out[name] = (p0, p1, np.where(exposed, y1, y0))

    # --- brand-lift survey ---------------------------------------------------
    surveyed = rng.random(n) < cfg.survey_rate
    if cfg.nonresponse == "mar":
        r = expit(-0.7 + 0.6 * eng + 0.7 * exposed - 0.4 * old + 0.3 * aff_z)
    else:
        r = np.full(n, 0.35)
    responded = surveyed & (rng.random(n) < r)

    users = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "campaign_id": camp["campaign_id"].to_numpy()[cidx],
        "age_group": np.array(AGE_GROUPS)[age],
        "region": np.array(REGIONS)[region],
        "device": np.where(mobile, "mobile", "desktop"),
        "sessions_30d": sessions,
        "log_sessions": np.round(log_sessions, 6),
        "affinity": np.round(affinity, 6),
        "income": income_obs,
        "prior_purchaser": prior.astype(int),
        "exposed": exposed.astype(int),
        "frequency": frequency.astype(int),
        "converted": out["converted"][2],
        "surveyed": surveyed.astype(int),
        "responded": responded.astype(int),
        "aware": np.where(responded, out["aware"][2], np.nan),
        "intent": np.where(responded, out["intent"][2], np.nan),
        "latent_intent": U,
        "true_propensity": e_true,
        "true_response_prob": np.where(surveyed, r, 0.0),
        "true_frequency_potential": freq_pot,
    })
    for name in ["converted", "aware", "intent"]:
        p0, p1, _ = out[name]
        users[f"true_p0_{name}"] = p0
        users[f"true_p1_{name}"] = p1
        users[f"true_effect_{name}"] = p1 - p0

    data = CampaignData(config=cfg, users=users, campaigns=camp)
    data.truth = true_effects(users)
    return data


def _calibrate_intercept(score: np.ndarray, target: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(mid + score).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def true_effects(users: pd.DataFrame, by: str | None = None) -> dict:
    """Exact ATT/ATE (expected-value scale) per outcome, optionally per group."""
    def calc(df):
        t = df["exposed"].to_numpy() == 1
        res = {}
        for o in ["converted", "aware", "intent"]:
            eff = df[f"true_effect_{o}"].to_numpy()
            p0 = df[f"true_p0_{o}"].to_numpy()
            res[o] = {
                "ATT": float(eff[t].mean()) if t.any() else np.nan,
                "ATE": float(eff.mean()),
                "baseline_ATT": float(p0[t].mean()) if t.any() else np.nan,
                "relative_ATT": float(eff[t].mean() / p0[t].mean()) if t.any() else np.nan,
            }
        return res

    if by is None:
        return calc(users)
    return {k: calc(g) for k, g in users.groupby(by)}


def true_frequency_curve(users: pd.DataFrame, outcome: str = "converted", max_f: int = 10) -> pd.DataFrame:
    """Average effect among exposed users at each realised frequency (exact)."""
    ex = users[users["exposed"] == 1]
    g = ex.groupby("frequency")[f"true_effect_{outcome}"].agg(["mean", "size"]).reset_index()
    return g[g["frequency"] <= max_f].rename(columns={"mean": "true_effect", "size": "n"})


# ---------------------------------------------------------------------------
# Raw event tables (what a measurement team actually receives)
# ---------------------------------------------------------------------------


def to_raw_tables(data: CampaignData, inject_issues: bool = True, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Explode the user frame into event-level tables, optionally with dirty rows.

    Tables: users, campaigns, user_campaign (eligible audience), impressions,
    conversions, survey_invites, survey_responses, campaign_spend_daily. Prior purchasers get a conversion event in the 90 days
    before launch, so ``prior_purchaser`` must be *derived* in SQL rather than
    read off a column. Injected issues (duplicate impressions, impressions
    outside the flight window, conversions with NULL user ids, survey answers
    from unknown users, an out-of-domain survey answer) must be caught by the
    pipeline's quality checks.
    """
    rng = np.random.default_rng(50_000 + seed)
    u = data.users
    camp = data.campaigns.set_index("campaign_id")
    sec = lambda x: pd.to_timedelta(x, unit="s")  # noqa: E731

    users = u[["user_id", "age_group", "region", "device", "sessions_30d", "affinity", "income"]].copy()
    users = users.rename(columns={"income": "income_score"})

    # impressions
    ex = u[u["frequency"] > 0]
    rep = np.repeat(ex.index.to_numpy(), ex["frequency"].to_numpy())
    imp = pd.DataFrame({"user_id": u.loc[rep, "user_id"].to_numpy(), "campaign_id": u.loc[rep, "campaign_id"].to_numpy()})
    st = camp.loc[imp["campaign_id"], "start_date"].to_numpy()
    imp["ts"] = st + sec(rng.uniform(0, 27 * 86400, len(imp)))
    fmt = camp.loc[imp["campaign_id"], "format"].map(CPM).to_numpy()
    imp["cost_usd"] = np.round(fmt / 1000 * rng.uniform(0.7, 1.3, len(imp)), 6)
    imp.insert(0, "impression_id", np.arange(1, len(imp) + 1))

    # conversions in window (after the user's first impression if exposed)
    first = imp.groupby("user_id")["ts"].min()
    conv_users = u[u["converted"] == 1]
    cs = camp.loc[conv_users["campaign_id"], "start_date"].to_numpy()
    lo = np.where(conv_users["exposed"] == 1, first.reindex(conv_users["user_id"]).to_numpy(), cs)
    lo = pd.to_datetime(lo)
    end = pd.to_datetime(cs) + pd.Timedelta(days=27, hours=23)
    span = (end - lo).total_seconds().to_numpy()
    conv = pd.DataFrame({"user_id": conv_users["user_id"].to_numpy(),
                         "ts": lo + sec(rng.uniform(0, 1, len(conv_users)) * np.maximum(span, 1)),
                         "order_value_usd": np.round(rng.gamma(2.0, 30.0, len(conv_users)), 2)})
    # prior purchases (pre-period)
    pp = u[u["prior_purchaser"] == 1]
    pstart = pd.to_datetime(camp.loc[pp["campaign_id"], "start_date"].to_numpy())
    prior = pd.DataFrame({"user_id": pp["user_id"].to_numpy(),
                          "ts": pstart - sec(rng.uniform(86400, 89 * 86400, len(pp))),
                          "order_value_usd": np.round(rng.gamma(2.0, 30.0, len(pp)), 2)})
    conversions = pd.concat([prior, conv], ignore_index=True)

    # survey (long format)
    resp = u[u["responded"] == 1]
    rend = pd.to_datetime(camp.loc[resp["campaign_id"], "end_date"].to_numpy())
    rts = rend + sec(rng.uniform(3600, 6 * 86400, len(resp)))
    survey = pd.concat([
        pd.DataFrame({"user_id": resp["user_id"].to_numpy(), "campaign_id": resp["campaign_id"].to_numpy(),
                      "question": q, "answer": resp[q].astype(int).to_numpy(), "ts": rts})
        for q in ["aware", "intent"]], ignore_index=True)
    invites = u[u["surveyed"] == 1][["user_id", "campaign_id"]].copy()

    # daily spend per campaign
    imp_day = imp.assign(day=imp["ts"].dt.floor("D")).groupby(["campaign_id", "day"], as_index=False)["cost_usd"].sum()
    spend = imp_day.rename(columns={"day": "date", "cost_usd": "spend_usd"})

    campaigns = data.campaigns.copy()

    if inject_issues:
        dup = imp.sample(frac=0.005, random_state=seed)
        oow = imp.sample(n=max(5, len(imp) // 500), random_state=seed + 1).copy()
        oow["ts"] = oow["ts"] + pd.Timedelta(days=45)
        oow["impression_id"] = np.arange(len(imp) + 1, len(imp) + 1 + len(oow))
        imp = pd.concat([imp, dup, oow], ignore_index=True)
        bad_conv = pd.DataFrame({"user_id": [None] * 25, "ts": conv["ts"].iloc[:25].to_numpy(),
                                 "order_value_usd": 10.0})
        conversions = pd.concat([conversions, bad_conv], ignore_index=True)
        ghost = survey.iloc[:30].copy()
        ghost["user_id"] = ghost["user_id"] + 10_000_000
        bad_ans = survey.iloc[30:33].copy()
        bad_ans["answer"] = 7
        survey = pd.concat([survey, ghost, bad_ans], ignore_index=True)

    conversions["user_id"] = conversions["user_id"].astype("Int64")
    audience = u[["user_id", "campaign_id"]].copy()  # eligible audience list per campaign
    return {"users": users, "campaigns": campaigns, "user_campaign": audience, "impressions": imp,
            "conversions": conversions,
            "survey_invites": invites, "survey_responses": survey, "campaign_spend_daily": spend}


def design_frame(users: pd.DataFrame) -> pd.DataFrame:
    """Analyst-facing covariates (for estimators): the observed columns only."""
    cols = ["user_id", "campaign_id", "age_group", "region", "device", "log_sessions", "affinity",
            "income", "prior_purchaser", "exposed", "frequency", "converted", "surveyed", "responded",
            "aware", "intent"]
    out = users[cols].copy()
    out["affinity_logit"] = logit_affinity(out)
    return out


def logit_affinity(df: pd.DataFrame) -> np.ndarray:
    return logit(np.clip(df["affinity"].to_numpy(float), 1e-6, 1 - 1e-6))
