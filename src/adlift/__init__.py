"""adlift: advertising incrementality, brand lift and marketing mix modeling.

Every estimator in this package is scored against a known answer before it is
trusted on data where the answer is unknown. Known answers come from:

* simulators that return both potential outcomes per user (``y0``, ``y1``) and,
  for the marketing mix model, the exact channel contributions; and
* real randomized ad experiments (Criteo uplift, ads-vs-PSA), where the
  randomized contrast is the benchmark that observational methods must match.
"""

from adlift.results import EffectEstimate

__all__ = ["EffectEstimate"]
__version__ = "0.1.0"
