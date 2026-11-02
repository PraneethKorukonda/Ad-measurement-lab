"""Turn a user table into a numeric design matrix.

Parametric estimators need explicit encoding decisions, and those decisions are
part of the model: which covariates enter, how missing values are handled, and
whether nonlinear terms are included. Missing numeric values are mean-imputed
with an added missing-indicator column, so that "covariate unknown" can itself
predict exposure (it often does: logged-out users are both harder to target and
less likely to convert).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Design:
    X: np.ndarray
    names: list[str]

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape


def make_design(
    df: pd.DataFrame,
    numeric: list[str] | tuple[str, ...] = (),
    categorical: list[str] | tuple[str, ...] = (),
    squares: list[str] | tuple[str, ...] = (),
    interactions: list[tuple[str, str]] | tuple = (),
    standardize: bool = True,
) -> Design:
    """Encode covariates; no intercept column (estimators add their own).

    ``squares`` and ``interactions`` let the analyst add the nonlinear terms a
    correctly specified model would need. Benchmarks deliberately omit them to
    show what misspecification does.
    """
    cols: list[np.ndarray] = []
    names: list[str] = []
    num = {}
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        miss = np.isnan(v)
        if miss.any():
            fill = np.nanmean(v) if (~miss).any() else 0.0
            v = np.where(miss, fill, v)
            cols.append(miss.astype(float))
            names.append(f"{c}__missing")
        if standardize:
            sd = v.std()
            v = (v - v.mean()) / (sd if sd > 0 else 1.0)
        num[c] = v
        cols.append(v)
        names.append(c)
    for c in squares:
        v = num[c]
        cols.append(v**2 - np.mean(v**2))
        names.append(f"{c}^2")
    for a, b in interactions:
        va = num[a] if a in num else df[a].to_numpy(float)
        vb = num[b] if b in num else df[b].to_numpy(float)
        cols.append(va * vb)
        names.append(f"{a}*{b}")
    for c in categorical:
        s = df[c].astype("object").where(df[c].notna(), "__missing__").astype(str)
        levels = sorted(s.unique())
        for lev in levels[1:]:
            cols.append((s == lev).to_numpy(float))
            names.append(f"{c}={lev}")
    X = np.column_stack(cols) if cols else np.zeros((len(df), 0))
    return Design(X=X, names=names)
