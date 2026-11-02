"""Simulator truth, SQL pipeline parity and data-quality checks."""

import numpy as np
import pandas as pd

from adlift.pipeline import diagnostics as dg
from adlift.pipeline.runner import build_campaign_tables
from adlift.sim.campaign import (
    SCENARIOS,
    design_frame,
    scenario_config,
    simulate_campaign,
    to_raw_tables,
    true_effects,
)


def test_scenarios_all_simulate():
    for name in SCENARIOS:
        d = simulate_campaign(scenario_config(name, n_users=3000, seed=1))
        assert 0.3 < d.users["exposed"].mean() < 0.5
        assert set(d.truth) == {"converted", "aware", "intent"}


def test_null_scenario_has_zero_truth():
    d = simulate_campaign(scenario_config("null_effect", n_users=3000, seed=2))
    assert all(abs(v["ATT"]) < 1e-12 for v in d.truth.values())


def test_randomized_propensity_constant_and_targeted_not():
    r = simulate_campaign(scenario_config("randomized", n_users=5000, seed=3))
    s = simulate_campaign(scenario_config("targeted", n_users=5000, seed=3))
    assert r.users["true_propensity"].std() < 1e-12
    assert s.users["true_propensity"].std() > 0.1


def test_observed_outcome_matches_potential_outcome_probabilities():
    d = simulate_campaign(scenario_config("targeted", n_users=60000, seed=4))
    u = d.users
    ex = u[u.exposed == 1]
    assert abs(ex["converted"].mean() - ex["true_p1_converted"].mean()) < 0.01


def test_segment_truth_aggregates_to_overall():
    d = simulate_campaign(scenario_config("targeted", n_users=8000, seed=5))
    seg = true_effects(d.users, by="device")
    n1 = d.users.groupby("device")["exposed"].sum()
    w = np.array([n1[k] for k in seg]) / n1.sum()
    agg = sum(w_i * seg[k]["converted"]["ATT"] for w_i, k in zip(w, seg))
    assert abs(agg - d.truth["converted"]["ATT"]) < 1e-10


def test_sql_pipeline_reproduces_simulator_exactly():
    d = simulate_campaign(scenario_config("survey_nonresponse", n_users=6000, seed=6))
    res = build_campaign_tables(to_raw_tables(d, inject_issues=True))
    a = res.analysis.set_index("user_id").sort_index()
    b = design_frame(d.users).set_index("user_id").sort_index()
    for c in ["exposed", "frequency", "converted", "surveyed", "responded", "prior_purchaser"]:
        assert (a[c].astype(float).to_numpy() == b[c].astype(float).to_numpy()).all(), c
    for c in ["aware", "intent", "log_sessions", "affinity_logit", "income"]:
        assert np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True, atol=1e-5), c
    assert res.passed


def test_quality_checks_flag_injected_issues_and_pass_clean_data():
    d = simulate_campaign(scenario_config("targeted", n_users=4000, seed=7))
    dirty = build_campaign_tables(to_raw_tables(d, inject_issues=True)).raw_quality.set_index("check_name")
    clean = build_campaign_tables(to_raw_tables(d, inject_issues=False)).raw_quality.set_index("check_name")
    for chk in ["duplicate_impression_rows", "impressions_outside_flight", "conversions_null_user",
                "survey_from_uninvited_or_unknown", "survey_answer_out_of_domain"]:
        assert dirty.loc[chk, "failing"] > 0, chk
        assert clean.loc[chk, "failing"] == 0, chk


def test_diagnostics():
    d = simulate_campaign(scenario_config("survey_nonresponse", n_users=20000, seed=8))
    df = design_frame(d.users)
    miss = dg.missingness_report(df, ["income"], by=["device"])
    assert miss.loc[0, "missing_rate[device=mobile]"] > miss.loc[0, "missing_rate[device=desktop]"]
    assert miss.loc[0, "chi2_p[device]"] < 0.001  # MAR, not MCAR
    sv = dg.survey_response_diagnostics(df, ["log_sessions"])
    assert sv["response_rate_exposed"] > sv["response_rate_unexposed"]
    srm = dg.srm_test(5000, 5000, 0.5)
    assert srm["p_value"] > 0.99
    t = df.exposed.to_numpy()
    x = df[["log_sessions"]].to_numpy()
    bal = dg.balance_table(x, ["log_sessions"], t, weights=dg.att_weights(d.users.true_propensity.to_numpy(), t))
    assert abs(bal.loc[0, "smd_unweighted"]) > 0.2 and abs(bal.loc[0, "smd_weighted"]) < 0.05


def test_raw_tables_shapes():
    d = simulate_campaign(scenario_config("targeted", n_users=2000, seed=9))
    raw = to_raw_tables(d, inject_issues=False)
    assert len(raw["impressions"]) == int(d.users["frequency"].sum())
    assert isinstance(raw["survey_responses"], pd.DataFrame)
