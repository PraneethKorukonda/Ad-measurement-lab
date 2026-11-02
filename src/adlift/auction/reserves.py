"""Reserve-price (floor) policies.

A floor trades fill for price. In a second-price auction the winner pays
max(second bid, floor), so a floor only earns money when it lands between the
top two bids, and only costs money when it exceeds the top bid (the slot goes
unfilled). Thin markets, where the gap between the top two bids is large, are
where a floor pays.

Policies here:

* ``none``          - no floor.
* ``fixed``         - one global floor (the "status quo" in the experiments).
* ``myerson``       - Myerson's optimal reserve per segment, r = (1 - F(r)) / f(r),
                      with F a lognormal fitted to logged bids. Optimal for
                      symmetric bidders with regular value distributions; our
                      bidders are asymmetric, budget-paced and only sometimes
                      present, so this is a model-based approximation.
* ``empirical``     - per-segment floor chosen by replaying logged bids on a
                      training day and picking the revenue-maximising grid point.
                      Model-free, but blind to how bidders will react.
* ``empirical_fill``- the same, subject to a minimum fill rate per segment
                      (a common business guardrail: unfilled slots are lost
                      reach for advertisers and lost data for the platform).

Fitting reserves on one day and evaluating them on *other* days avoids
overfitting the floor to one day's noise.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from adlift.auction.market import SEGMENTS


def fixed(cpm: float) -> np.ndarray:
    return np.full(len(SEGMENTS), cpm / 1000)


def none() -> np.ndarray:
    return np.zeros(len(SEGMENTS))


def _top2(bids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.where(np.isnan(bids), -np.inf, bids)
    m = b.shape[1]
    t2 = np.partition(b, m - 2, axis=1)[:, -2:]
    return t2[:, 1], t2[:, 0]


def replay_revenue(bids: np.ndarray, seg: np.ndarray, reserve_by_segment: np.ndarray) -> tuple[float, float]:
    """Offline counterfactual on logged second-price bids (assumes bids do not react)."""
    top, second = _top2(bids)
    r = reserve_by_segment[seg]
    filled = top >= r
    price = np.where(filled, np.maximum(np.where(np.isfinite(second), second, 0.0), r), 0.0)
    return float(price.sum()), float(filled.mean())


def myerson(bids: np.ndarray, seg: np.ndarray, min_n: int = 200) -> np.ndarray:
    """Per-segment Myerson reserve from a lognormal fit to all logged bids."""
    out = np.zeros(len(SEGMENTS))
    for s in range(len(SEGMENTS)):
        b = bids[seg == s]
        b = b[np.isfinite(b) & (b > 0)]
        if len(b) < min_n:
            continue
        mu, sd = np.log(b).mean(), np.log(b).std()
        dist = stats.lognorm(s=sd, scale=np.exp(mu))

        def virtual(r, dist=dist):
            return r - dist.sf(r) / dist.pdf(r)

        lo, hi = dist.ppf(0.01), dist.ppf(0.99)
        out[s] = optimize.brentq(virtual, lo, hi) if virtual(lo) < 0 < virtual(hi) else 0.0
    return out


def empirical(bids: np.ndarray, seg: np.ndarray, min_fill: float = 0.0, n_grid: int = 60) -> np.ndarray:
    """Revenue-maximising per-segment floor on logged bids, optionally fill-constrained."""
    top, second = _top2(bids)
    out = np.zeros(len(SEGMENTS))
    for s in range(len(SEGMENTS)):
        m = seg == s
        t, sec = top[m], second[m]
        t_ok = t[np.isfinite(t)]
        if len(t_ok) < 50:
            continue
        sec = np.where(np.isfinite(sec), sec, 0.0)
        grid = np.concatenate([[0.0], np.quantile(t_ok, np.linspace(0.0, 0.9, n_grid))])
        best, best_r = -1.0, 0.0
        for r in grid:
            f = t >= r
            if f.mean() < min_fill:
                continue
            rev = np.sum(np.where(f, np.maximum(sec, r), 0.0))
            if rev > best:
                best, best_r = rev, r
        out[s] = best_r
    return out
