"""A simulated video-ad marketplace: opportunities, advertisers, auctions, pacing and shading.

Each day has ~100K ad opportunities (pre-roll or mid-roll slots), each in a
segment defined by viewer country tier, content category and device.
Advertisers have private per-impression values (expected conversion value),
budgets, targeting and creative quality. The platform runs a sealed-bid
auction per opportunity, with a reserve price (floor) set by policy.

What makes this more than a textbook auction:

* **Thin markets.** Not every eligible advertiser bids on every slot, so some
  segments (tier-3 news, say) have one or two bidders. Reserves matter most
  there, because without a floor the second-highest bid can be near zero.
* **Budgets and pacing.** Advertisers have daily budgets and a pacing
  controller that throttles bids hour by hour. Raising floors changes spend,
  so advertisers re-pace: the market reacts, and an offline replay of logged
  bids cannot see that reaction.
* **First-price bid shading.** Under first-price rules, bidders shade bids
  below value using the bid landscape they observe (Google Ad Manager moved to
  a unified first-price auction in 2019). Shading responds to the floor.
* **Creative quality.** Low-value advertisers tend to have lower-quality
  creatives. Low floors fill more slots, partly with ads viewers dislike, which
  is the link to the viewer-experience study.

Money is in dollars per impression; ``RPM`` = revenue per 1000 opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIERS = ["T1", "T2", "T3"]
CONTENT = ["beauty", "education", "gaming", "music", "news"]
DEVICES = ["ctv", "desktop", "mobile"]
TIER_CPM = {"T1": 9.0, "T2": 3.5, "T3": 1.2}  # median value CPM by country tier
CONTENT_MULT = {"beauty": 1.25, "education": 1.1, "gaming": 0.9, "music": 0.8, "news": 0.75}
DEVICE_MULT = {"ctv": 1.35, "desktop": 1.05, "mobile": 0.9}


@dataclass
class Market:
    """Advertisers are fixed across days; opportunities are drawn per day."""

    advertisers: pd.DataFrame
    eligibility: np.ndarray  # (n_adv, n_tier, n_content) bool
    seed: int
    value_sigma: float = 0.8
    bid_prob: float = 0.15

    @property
    def n_adv(self) -> int:
        return len(self.advertisers)


def make_market(n_advertisers: int = 60, seed: int = 0, bid_prob: float = 0.15,
                budget_range: tuple[float, float] = (0.35, 1.6)) -> Market:
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 0.45, n_advertisers)  # advertiser value level (log scale)
    quality = 0.45 * (mu / mu.std()) + np.sqrt(1 - 0.45**2) * rng.normal(0, 1, n_advertisers)
    elig = np.zeros((n_advertisers, len(TIERS), len(CONTENT)), bool)
    for a in range(n_advertisers):
        k = rng.integers(1, 4)
        contents = rng.choice(len(CONTENT), size=k, replace=False, p=[0.25, 0.2, 0.25, 0.2, 0.1])
        tiers = [0] if rng.random() < 0.3 else ([0, 1] if rng.random() < 0.5 else [0, 1, 2])
        for c in contents:
            elig[a, tiers, c] = True
    adv = pd.DataFrame({
        "advertiser_id": [f"A{a:02d}" for a in range(n_advertisers)],
        "value_level": mu,
        "creative_quality": quality,
        "daily_budget": np.full(n_advertisers, 1e9),
        "n_segments_targeted": elig.sum(axis=(1, 2)),
    })
    market = Market(advertisers=adv, eligibility=elig, seed=seed, bid_prob=bid_prob)
    # Budgets: a multiple of each advertiser's unconstrained spend on a calibration
    # day (no floor). Multiples below 1 make the budget bind, so pacing matters.
    cal = draw_day(market, day=-1)
    unconstrained = simulate_day(market, cal, np.zeros(len(TIERS) * len(CONTENT)), pacing=False).spend
    mult = rng.uniform(*budget_range, n_advertisers)
    market.advertisers["daily_budget"] = np.round(np.maximum(unconstrained, 1.0) * mult, 2)
    market.advertisers["budget_multiple"] = mult
    return market


@dataclass
class Day:
    """One day of opportunities with the full (n_opp x n_adv) value matrix (NaN = not bidding)."""

    opportunities: pd.DataFrame
    values: np.ndarray
    seg: np.ndarray  # tier*len(CONTENT)+content, 15 segments
    hour: np.ndarray


def draw_day(market: Market, n_opportunities: int = 100_000, day: int = 0,
             value_scale: float = 1.0) -> Day:
    rng = np.random.default_rng(1_000 * (market.seed + 1) + day)
    n = n_opportunities
    tier = rng.choice(3, n, p=[0.35, 0.35, 0.30])
    content = rng.choice(5, n, p=[0.15, 0.2, 0.3, 0.25, 0.1])
    device = rng.choice(3, n, p=[0.2, 0.2, 0.6])
    hour = np.sort(rng.integers(0, 24, n))
    base = (np.array([TIER_CPM[t] for t in TIERS])[tier]
            * np.array([CONTENT_MULT[c] for c in CONTENT])[content]
            * np.array([DEVICE_MULT[d] for d in DEVICES])[device]) / 1000
    mu = market.advertisers["value_level"].to_numpy()
    eligible = market.eligibility[:, tier, content].T  # (n, n_adv)
    participates = eligible & (rng.random(eligible.shape) < market.bid_prob)
    logv = np.log(base)[:, None] + mu[None, :] + rng.normal(0, market.value_sigma, eligible.shape)
    values = np.where(participates, np.exp(logv) * value_scale, np.nan)
    opp = pd.DataFrame({
        "opportunity_id": np.arange(n),
        "tier": np.array(TIERS)[tier],
        "content": np.array(CONTENT)[content],
        "device": np.array(DEVICES)[device],
        "hour": hour,
        "segment": np.array(TIERS)[tier].astype(object) + "/" + np.array(CONTENT)[content],
        "n_bidders": participates.sum(axis=1),
    })
    return Day(opportunities=opp, values=values, seg=tier * 5 + content, hour=hour)


SEGMENTS = [f"{t}/{c}" for t in TIERS for c in CONTENT]


# ---------------------------------------------------------------------------
# Auction mechanics
# ---------------------------------------------------------------------------


def clear(bids: np.ndarray, reserve: np.ndarray, fmt: str = "second") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one sealed-bid auction per row. NaN bids mean "not bidding".

    Returns (winner index or -1, price, filled). Second price: winner pays
    max(second-highest bid, reserve). First price: winner pays own bid.
    """
    b = np.where(np.isnan(bids), -np.inf, bids)
    n, m = b.shape
    if m >= 2:
        top2 = np.partition(b, m - 2, axis=1)[:, -2:]
        second = top2[:, 0]
    else:
        second = np.full(n, -np.inf)
    winner = np.argmax(b, axis=1)
    top = b[np.arange(n), winner]
    filled = top >= reserve
    if fmt == "second":
        price = np.maximum(np.where(np.isfinite(second), second, 0.0), reserve)
    elif fmt == "first":
        price = top
    else:  # pragma: no cover
        raise ValueError(fmt)
    price = np.where(filled, price, 0.0)
    return np.where(filled, winner, -1), price, filled


@dataclass
class DayResult:
    winner: np.ndarray
    price: np.ndarray
    filled: np.ndarray
    winner_value: np.ndarray
    winner_quality: np.ndarray
    bids: np.ndarray  # final submitted bids (for logging / replay)
    spend: np.ndarray  # per advertiser
    pacing: np.ndarray  # final pacing multipliers
    meta: dict = field(default_factory=dict)


def simulate_day(market: Market, day: Day, reserve_by_segment: np.ndarray, fmt: str = "second",
                 shading: np.ndarray | None = None, pacing: bool = True,
                 pacing_rate: float = 0.6) -> DayResult:
    """Hour-by-hour forward simulation with budget pacing (the "online" outcome).

    Pacing: each advertiser bids ``alpha * b(v)``, where ``alpha`` starts at 1
    and is multiplied by exp(rate * (target - actual) / target) each hour, with
    the target being an even spend of the remaining budget over remaining hours.
    Advertisers that exhaust their budget stop bidding.
    """
    n, A = day.values.shape
    budget = market.advertisers["daily_budget"].to_numpy().astype(float)
    remaining = budget.copy()
    alpha = np.ones(A)
    # shading: None (bid value), (A,) one factor per advertiser, or (A, 3) per advertiser x tier
    if shading is None:
        S = np.ones((n, A))
    elif np.ndim(shading) == 1:
        S = np.broadcast_to(shading, (n, A))
    else:
        S = shading[:, day.seg // 5].T
    reserve = reserve_by_segment[day.seg]
    winner = np.full(n, -1)
    price = np.zeros(n)
    filled = np.zeros(n, bool)
    bids_all = np.full_like(day.values, np.nan)
    for h in range(24):
        idx = np.flatnonzero(day.hour == h)
        if len(idx) == 0:
            continue
        v = day.values[idx]
        active = remaining > 0
        bids = v * alpha[None, :] * S[idx]
        bids[:, ~active] = np.nan
        w, p, f = clear(bids, reserve[idx], fmt)
        # budget cap: drop wins that would overspend (advertiser goes dark mid-hour)
        spend_h = np.bincount(w[f], weights=p[f], minlength=A)
        over = spend_h > remaining
        if over.any():
            for a in np.flatnonzero(over):
                rows = np.flatnonzero(w == a)
                cum = np.cumsum(p[rows])
                drop = rows[cum > remaining[a]]
                f[drop], p[drop], w[drop] = False, 0.0, -1
            spend_h = np.bincount(w[f], weights=p[f], minlength=A)
        remaining -= spend_h
        winner[idx], price[idx], filled[idx] = w, p, f
        bids_all[idx] = bids
        if pacing and h < 23:
            hours_left = 23 - h
            target = np.maximum(remaining, 0) / hours_left
            ratio = (target - spend_h) / np.maximum(target, 1e-9)
            alpha = np.clip(alpha * np.exp(pacing_rate * np.clip(ratio, -1, 1)), 0.05, 1.0)
    vals = np.where(filled, day.values[np.arange(n), np.maximum(winner, 0)], 0.0)
    qual = np.where(filled, market.advertisers["creative_quality"].to_numpy()[np.maximum(winner, 0)], np.nan)
    return DayResult(winner=winner, price=price, filled=filled, winner_value=vals, winner_quality=qual,
                     bids=bids_all, spend=budget - remaining, pacing=alpha,
                     meta={"budget": budget, "fmt": fmt})


def best_response_shading(market: Market, day: Day, reserve_by_segment: np.ndarray,
                          grid: np.ndarray | None = None, rounds: int = 4) -> np.ndarray:
    """First-price shading factors (advertiser x country tier) by iterated best response.

    Each advertiser picks, separately for each tier, the factor s in ``grid``
    that maximises its surplus sum(v - s*v) over the slots it would win against
    the other bidders' current shaded bids and the floor. Context-specific
    shading matters: a bidder facing four rivals in T1 should shade far less
    than one facing a lone rival in T3. A few rounds approximate an equilibrium.
    Budgets are ignored here (online pacing handles them).
    """
    grid = np.linspace(0.3, 1.0, 36) if grid is None else grid
    v = np.where(np.isnan(day.values), -np.inf, day.values)
    reserve = reserve_by_segment[day.seg]
    tier = day.seg // 5
    A = v.shape[1]
    s = np.full((A, 3), 0.8)
    for _ in range(rounds):
        new = s.copy()
        b = v * s[:, tier].T
        for a in range(A):
            others = np.delete(b, a, axis=1)
            comp_all = np.maximum(others.max(axis=1), reserve)
            for k in range(3):
                rows = np.isfinite(v[:, a]) & (tier == k)
                if not rows.any():
                    continue
                va, comp = v[rows, a], comp_all[rows]
                win = (grid[:, None] * va[None, :]) >= comp[None, :]
                surplus = (win * (va[None, :] * (1 - grid[:, None]))).sum(axis=1)
                new[a, k] = grid[int(np.argmax(surplus))]
        s = 0.5 * s + 0.5 * new  # damped update for stability
    return s


def summarize(market: Market, day: Day, res: DayResult) -> dict:
    n = len(res.price)
    rev = res.price.sum()
    val = res.winner_value.sum()
    winners = res.winner[res.filled]
    shares = np.bincount(winners, minlength=market.n_adv) / max(len(winners), 1)
    budget = res.meta["budget"]
    return {
        "rpm": 1000 * rev / n,
        "fill_rate": res.filled.mean(),
        "advertiser_surplus_per_1k": 1000 * (val - rev) / n,
        "advertiser_roi": val / rev if rev > 0 else np.nan,
        "welfare_per_1k": 1000 * val / n,
        "avg_winner_quality": float(np.nanmean(res.winner_quality)),
        "winner_hhi": float(np.sum(shares**2)),
        "budget_utilization": float(res.spend.sum() / budget.sum()),
        "n_advertisers_winning": int((shares > 0).sum()),
    }
