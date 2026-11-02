"""Media transformations shared by the simulator and the PyMC model.

Adstock (carryover): spend this week keeps working for several weeks. We use
geometric weights w_l = alpha**l normalised to sum to 1 over L lags, so a
constant spend level x has steady-state adstock exactly x. That keeps the
saturation curve's half-saturation point interpretable in spend units.

Saturation (diminishing returns): the Hill curve s**h / (s**h + K**h), where s is
adstocked spend divided by the channel's average spend. K is the
half-saturation point (in multiples of average spend) and h the shape. h > 1
gives an S-curve; h <= 1 is concave everywhere. This is the specification of
Jin, Wang, Sun, Chan and Koehler (2017), "Bayesian Methods for Media Mix
Modeling with Carryover and Shape Effects" (Google).
"""

from __future__ import annotations

import numpy as np


def lag_matrix(x: np.ndarray, L: int) -> np.ndarray:
    """(T + L - 1, C) spend with L-1 weeks of history -> (T, L, C) lag tensor.

    Row t, lag l holds spend from week t - l. The first L-1 rows of ``x`` are
    pre-sample history used only to initialise carryover.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    T = x.shape[0] - (L - 1)
    out = np.empty((T, L, x.shape[1]))
    for lag in range(L):
        out[:, lag, :] = x[L - 1 - lag : L - 1 - lag + T]
    return out


def adstock_weights(alpha: np.ndarray, L: int) -> np.ndarray:
    alpha = np.atleast_1d(alpha)
    w = alpha[None, :] ** np.arange(L)[:, None]
    return w / w.sum(axis=0, keepdims=True)


def adstock(x_lag: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    L = x_lag.shape[1]
    return np.einsum("tlc,lc->tc", x_lag, adstock_weights(alpha, L))


def hill(s: np.ndarray, K: np.ndarray, h: np.ndarray) -> np.ndarray:
    s = np.maximum(s, 0.0)
    return s**h / (s**h + K**h)


def response(x_lag: np.ndarray, scale: np.ndarray, alpha, K, h, beta) -> np.ndarray:
    """Weekly contribution per channel (T, C) in revenue units."""
    return beta * hill(adstock(x_lag, alpha) / scale, K, h)
