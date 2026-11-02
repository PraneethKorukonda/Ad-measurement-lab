"""Auction mechanics, reserve policies, pacing and the viewer experiment."""

import numpy as np
from scipy import stats

from adlift.auction import reserves as R
from adlift.auction.market import clear, draw_day, make_market, simulate_day


def test_second_and_first_price_clearing():
    bids = np.array([[5.0, 3.0, np.nan], [2.0, np.nan, np.nan], [1.0, 4.0, 6.0]])
    reserve = np.array([1.0, 2.5, 7.0])
    w, p, f = clear(bids, reserve, "second")
    assert w.tolist() == [0, -1, -1] and f.tolist() == [True, False, False]
    assert p[0] == 3.0
    w, p, f = clear(bids, np.array([4.0, 0.0, 0.0]), "second")
    assert p[0] == 4.0 and p[1] == 0.0 and p[2] == 4.0  # floor binds; single bidder pays floor (0)
    w, p, f = clear(bids, np.zeros(3), "first")
    assert p.tolist() == [5.0, 2.0, 6.0]


def test_myerson_reserve_matches_numeric_optimum_for_lognormal():
    rng = np.random.default_rng(0)
    vals = rng.lognormal(0.0, 0.6, 200_000)
    bids = vals[:, None]
    seg = np.zeros(len(vals), int)
    r = R.myerson(bids, seg)[0]
    grid = np.linspace(0.2, 3, 400)
    dist = stats.lognorm(s=0.6)
    best = grid[np.argmax(grid * dist.sf(grid))]  # single-bidder revenue r * P(v > r)
    assert abs(r - best) < 0.05


def test_floor_raises_revenue_in_thin_markets_on_replay():
    m = make_market(seed=1)
    day = draw_day(m, 40_000, day=0)
    logged = simulate_day(m, day, R.none(), pacing=False)
    emp = R.empirical(logged.bids, day.seg)
    rev0, fill0 = R.replay_revenue(logged.bids, day.seg, R.none())
    rev1, fill1 = R.replay_revenue(logged.bids, day.seg, emp)
    assert rev1 > rev0 and fill1 < fill0


def test_pacing_never_overspends_budget():
    m = make_market(seed=2)
    day = draw_day(m, 50_000, day=3)
    res = simulate_day(m, day, R.fixed(1.0))
    assert np.all(res.spend <= m.advertisers["daily_budget"].to_numpy() + 1e-9)
    assert np.all(res.price[res.filled] >= 1.0 / 1000 - 1e-12)


def test_viewer_experiment_estimates_cover_truth():
    from adlift.auction.viewer import ViewerConfig, run_viewer_experiment

    r = run_viewer_experiment(ViewerConfig(n_viewers=8000, burn_in=4))
    w = r["weekly"]
    assert w["covers"].mean() > 0.75
    s = r["summary"].set_index(["arm", "metric"])
    assert s.loc[("C_double_pods", "revenue"), "week1_relative_pct"] > 20
    # long-term harm grows: watch time loss at week 6 exceeds week 1
    assert s.loc[("C_double_pods", "watch_minutes"), "week6_relative_pct"] < s.loc[("C_double_pods", "watch_minutes"), "week1_relative_pct"]
