"""A single result type shared by every estimator.

Keeping one return type means benchmark code can score any estimator the same
way (bias, RMSE, interval coverage) without special cases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats


@dataclass
class EffectEstimate:
    """Point estimate, standard error and interval for one causal estimand.

    ``estimand`` is "ATE", "ATT", "ITT", "TOT" (effect of treatment on the
    treated under noncompliance) or "ROAS"/"lift" for MMM quantities. Absolute
    effects are on the outcome scale (probability points for binary outcomes);
    ``relative`` is the effect divided by the counterfactual mean.
    """

    estimand: str
    method: str
    estimate: float
    se: float
    n: int
    alpha: float = 0.10
    baseline: float | None = None
    relative: float | None = None
    relative_se: float | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def z(self) -> float:
        return float(stats.norm.ppf(1 - self.alpha / 2))

    @property
    def ci(self) -> tuple[float, float]:
        return (self.estimate - self.z * self.se, self.estimate + self.z * self.se)

    @property
    def relative_ci(self) -> tuple[float, float] | None:
        if self.relative is None or self.relative_se is None:
            return None
        return (self.relative - self.z * self.relative_se, self.relative + self.z * self.relative_se)

    @property
    def pvalue(self) -> float:
        if not np.isfinite(self.se) or self.se <= 0:
            return float("nan")
        return float(2 * stats.norm.sf(abs(self.estimate / self.se)))

    def covers(self, truth: float) -> bool:
        lo, hi = self.ci
        return bool(lo <= truth <= hi)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci_low"], d["ci_high"] = self.ci
        d["pvalue"] = self.pvalue
        d.pop("diagnostics")
        return d

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lo, hi = self.ci
        s = f"{self.method} {self.estimand}: {self.estimate:+.5f} (SE {self.se:.5f}, {int((1 - self.alpha) * 100)}% CI [{lo:+.5f}, {hi:+.5f}])"
        if self.relative is not None:
            s += f", relative {self.relative:+.1%}"
        return s


def from_influence(
    estimand: str,
    method: str,
    estimate: float,
    influence: np.ndarray,
    *,
    baseline: float | None = None,
    baseline_influence: np.ndarray | None = None,
    alpha: float = 0.10,
    diagnostics: dict | None = None,
) -> EffectEstimate:
    """Build an estimate whose SE comes from a per-unit influence function.

    If the counterfactual mean (``baseline``) and its influence function are
    given, the relative lift tau / mu0 gets a delta-method SE:
    IF_rel = (IF_tau * mu0 - tau * IF_mu0) / mu0**2.
    """
    n = len(influence)
    se = float(np.std(influence, ddof=1) / np.sqrt(n))
    rel = rel_se = None
    if baseline is not None and baseline_influence is not None and baseline != 0:
        rel = estimate / baseline
        if_rel = (influence * baseline - estimate * baseline_influence) / baseline**2
        rel_se = float(np.std(if_rel, ddof=1) / np.sqrt(n))
    return EffectEstimate(
        estimand=estimand,
        method=method,
        estimate=float(estimate),
        se=se,
        n=n,
        alpha=alpha,
        baseline=None if baseline is None else float(baseline),
        relative=rel,
        relative_se=rel_se,
        diagnostics=diagnostics or {},
    )
