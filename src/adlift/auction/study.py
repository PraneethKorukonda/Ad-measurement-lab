"""Reserve-price study: which floor policy, judged on revenue, fill and advertiser outcomes.

Protocol (mirrors how a monetization team would evaluate a floor change):

1. Day 0 runs the status-quo policy (fixed $2 CPM floor). Its logged bids are the
   only data used to fit the candidate policies.
2. Each candidate is evaluated on held-out days 1..D two ways:
   * **offline replay** - re-clear that day's *logged* status-quo bids under the
     new floor. Cheap, but assumes bidders do not react.
   * **online** - re-run the day forward, letting pacing (and, for first-price,
     shading) respond to the new floor. This is the "truth" an A/B test on the
     auction would measure.
3. Differences vs. status quo are paired by day (same opportunities, same values),
   and intervals are t-intervals across days. Day-to-day variation, not
   per-impression noise, is the right unit of uncertainty for a daily metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from adlift.auction import reserves as R
from adlift.auction.market import (
    SEGMENTS,
    Market,
    best_response_shading,
    draw_day,
    make_market,
    simulate_day,
    summarize,
)

STATUS_QUO = "fixed_$2"
METRICS = ["rpm", "fill_rate", "advertiser_surplus_per_1k", "welfare_per_1k", "avg_winner_quality",
           "advertiser_roi", "budget_utilization"]


def fit_policies(market: Market, train_day) -> tuple[dict[str, np.ndarray], object]:
    logged = simulate_day(market, train_day, R.fixed(2.0))
    policies = {
        "no_floor": R.none(),
        STATUS_QUO: R.fixed(2.0),
        "fixed_$4": R.fixed(4.0),
        "myerson_segment": R.myerson(logged.bids, train_day.seg),
        "empirical_segment": R.empirical(logged.bids, train_day.seg),
        "empirical_fill85": R.empirical(logged.bids, train_day.seg, min_fill=0.85),
    }
    return policies, logged


def _paired_ci(x: np.ndarray, alpha: float = 0.10) -> tuple[float, float, float]:
    m = float(np.mean(x))
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(1 - alpha / 2, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def run_reserve_study(n_days: int = 14, n_opportunities: int = 100_000, seed: int = 0,
                      first_price: bool = True) -> dict[str, pd.DataFrame]:
    market = make_market(seed=seed)
    train = draw_day(market, n_opportunities, day=0)
    policies, _ = fit_policies(market, train)
    shading = {}
    if first_price:
        for k in [STATUS_QUO, "no_floor", "myerson_segment"]:
            shading[k] = best_response_shading(market, train, policies[k])

    rows, seg_rows, adv_rows = [], [], []
    for d in range(1, n_days + 1):
        day = draw_day(market, n_opportunities, day=d)
        logged = simulate_day(market, day, policies[STATUS_QUO])
        base_online = summarize(market, day, logged)
        n = len(day.seg)
        sq_seg = pd.Series(logged.price).groupby(day.seg).sum()
        for name, res_floor in policies.items():
            res = logged if name == STATUS_QUO else simulate_day(market, day, res_floor)
            online = summarize(market, day, res)
            rev_off, fill_off = R.replay_revenue(logged.bids, day.seg, res_floor)
            rows.append({"day": d, "format": "second", "policy": name, **online,
                         "offline_rpm": 1000 * rev_off / n, "offline_fill": fill_off,
                         "rpm_vs_sq": online["rpm"] - base_online["rpm"],
                         "offline_rpm_vs_sq": 1000 * rev_off / n - base_online["rpm"]})
            seg_rev = pd.Series(res.price).groupby(day.seg).sum()
            seg_fill = pd.Series(res.filled).groupby(day.seg).mean()
            for s in range(len(SEGMENTS)):
                seg_rows.append({"day": d, "policy": name, "segment": SEGMENTS[s],
                                 "revenue": seg_rev.get(s, 0.0), "revenue_sq": sq_seg.get(s, 0.0),
                                 "fill_rate": seg_fill.get(s, np.nan)})
            win = res.winner[res.filled]
            surplus = np.bincount(win, weights=(res.winner_value - res.price)[res.filled], minlength=market.n_adv)
            adv_rows.append(pd.DataFrame({"day": d, "policy": name, "advertiser_id": market.advertisers["advertiser_id"],
                                          "surplus": surplus, "spend": res.spend,
                                          "budget_multiple": market.advertisers["budget_multiple"]}))
        for name, sh in shading.items():
            res = simulate_day(market, day, policies[name], fmt="first", shading=sh)
            online = summarize(market, day, res)
            rows.append({"day": d, "format": "first", "policy": name, **online,
                         "offline_rpm": np.nan, "offline_fill": np.nan,
                         "rpm_vs_sq": online["rpm"] - base_online["rpm"], "offline_rpm_vs_sq": np.nan})

    daily = pd.DataFrame(rows)
    summary = []
    for (fmt, pol), g in daily.groupby(["format", "policy"], sort=False):
        rec = {"format": fmt, "policy": pol}
        for m in METRICS:
            rec[m] = g[m].mean()
        est, lo, hi = _paired_ci(g["rpm_vs_sq"].to_numpy())
        sq = daily[(daily.policy == STATUS_QUO) & (daily.format == "second")]["rpm"].mean()
        rec.update({"rpm_lift_pct": 100 * est / sq, "rpm_lift_ci_low_pct": 100 * lo / sq,
                    "rpm_lift_ci_high_pct": 100 * hi / sq})
        if fmt == "second":
            off, _, _ = _paired_ci(g["offline_rpm_vs_sq"].to_numpy())
            rec["offline_predicted_lift_pct"] = 100 * off / sq
        summary.append(rec)
    summary = pd.DataFrame(summary)

    seg = pd.DataFrame(seg_rows).groupby(["policy", "segment"], as_index=False).agg(
        revenue=("revenue", "sum"), revenue_sq=("revenue_sq", "sum"), fill_rate=("fill_rate", "mean"))
    seg["revenue_lift_pct"] = 100 * (seg["revenue"] / seg["revenue_sq"] - 1)
    nb = train.opportunities.groupby("segment")["n_bidders"].mean().rename("avg_bidders")
    seg = seg.merge(nb, left_on="segment", right_index=True)

    adv = pd.concat(adv_rows).groupby(["policy", "advertiser_id"], as_index=False).agg(
        surplus=("surplus", "sum"), spend=("spend", "sum"), budget_multiple=("budget_multiple", "first"))
    sq_adv = adv[adv.policy == STATUS_QUO].set_index("advertiser_id")["surplus"]
    adv["surplus_change_pct"] = 100 * (adv["surplus"] / adv["advertiser_id"].map(sq_adv) - 1)
    adv["budget_constrained"] = adv["budget_multiple"] < 1.0

    return {"daily": daily, "summary": summary, "segments": seg, "advertisers": adv,
            "policies": pd.DataFrame({k: 1000 * v for k, v in policies.items()}, index=SEGMENTS)}
