"""Bayesian marketing mix model in PyMC, with optional experiment calibration.

Model (weekly, revenue scaled by its mean, spend scaled by each channel's mean):

    revenue_t = intercept + trend * t + fourier_t' gamma + controls_t' delta
                + sum_c beta_c * Hill(Adstock_c(spend)_t; K_c, h_c) + noise_t

Priors are weakly informative and stated in interpretable units: K is the
half-saturation point in multiples of average spend, ``beta`` the maximum
weekly contribution as a share of average revenue. Posterior draws of the media
parameters give full distributions for ROAS (average return), marginal ROAS (the
return on the *next* dollar, which is what budget decisions need) and response
curves.

Why calibrate. An MMM is a regression on observational spend. If spend reacts
to demand (search budgets do, by design), the regression attributes demand to
the channel. No amount of Bayesian machinery fixes that, because the bias is in
the data, not the estimator. A geo experiment gives an unbiased estimate of one
channel's incremental revenue over a known window. ``lift_tests`` adds each
experiment as an extra observation: the model must reproduce the measured lift
(within its standard error) from the same parameters that explain the time
series. This is the approach used by Google Meridian (ROI priors from
experiments) and PyMC-Marketing (lift-test likelihood).

Sampling uses nutpie (a fast NUTS implementation); models are compiled once and
reused with new data, which makes Monte Carlo studies affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adlift.mmm.transforms import adstock, hill, lag_matrix

L_LAGS = 8
_COMPILED: dict = {}


def fourier(t: np.ndarray, period: float = 52.0, order: int = 2) -> np.ndarray:
    cols = []
    for k in range(1, order + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
    return np.column_stack(cols)


@dataclass
class LiftTest:
    """A geo experiment: channel spend multiplied by ``multiplier`` in test geos.

    ``multiplier=0`` is a go-dark holdout, ``1.5`` a +50% heavy-up.
    ``start``/``weeks`` index sample weeks; ``cooldown`` extra weeks count toward
    the measured lift (carryover). ``lift`` (revenue change caused in the test
    geos, negative for go-dark) and ``se`` are in revenue units.

    A go-dark test pins down the response curve between zero and current spend,
    i.e. *average* ROAS. A heavy-up pins down the slope just above current spend,
    i.e. *marginal* ROAS. Budget decisions need the second.
    """

    channel: str
    start: int
    weeks: int
    cooldown: int
    test_share: float
    multiplier: float
    lift: float
    se: float

    @classmethod
    def from_experiment(cls, e: dict) -> LiftTest:
        return cls(e["channel"], e["start"], e["weeks"], e["cooldown"], e["test_share"],
                   e["multiplier"], e["lift_estimate"], e["lift_se"])


@dataclass
class MMMInputs:
    channels: list[str]
    y: np.ndarray
    y_scale: float
    spend_scale: np.ndarray
    x_lag: np.ndarray  # (T, L, C) scaled
    spend: np.ndarray  # (T, C) raw sample spend
    spend_history: np.ndarray  # (T+L-1, C) raw
    fourier: np.ndarray
    controls: np.ndarray
    control_names: list[str]
    trend: np.ndarray
    lift: list = field(default_factory=list)
    scaling: dict = field(default_factory=dict)


def prepare(df: pd.DataFrame, channels: list[str], controls: list[str],
            spend_history: np.ndarray | None = None, lift_tests: list[LiftTest] | None = None,
            revenue_col: str = "revenue", spend_prefix: str = "spend_",
            scaling: dict | None = None) -> MMMInputs:
    """Build model inputs. Pass ``scaling`` from a training fit to score new weeks
    on the training scale (out-of-sample prediction)."""
    T = len(df)
    spend = df[[f"{spend_prefix}{c}" for c in channels]].to_numpy(float)
    if spend_history is None:  # no pre-sample history: pad by repeating the first weeks' mean
        pad = np.repeat(spend[:4].mean(0, keepdims=True), L_LAGS - 1, axis=0)
        spend_history = np.vstack([pad, spend])
    if scaling is None:
        scale = spend.mean(0)
        scale = np.where(scale > 0, scale, 1.0)
        Z0 = df[controls].to_numpy(float) if controls else np.zeros((T, 0))
        sd = Z0.std(0) if Z0.shape[1] else np.zeros(0)
        scaling = {"y_scale": float(df[revenue_col].mean()), "spend_scale": scale,
                   "z_mean": Z0.mean(0) if Z0.shape[1] else np.zeros(0),
                   "z_sd": np.where(sd > 0, sd, 1.0), "T_ref": T}
    scale, y_scale = scaling["spend_scale"], scaling["y_scale"]
    Z = df[controls].to_numpy(float) if controls else np.zeros((T, 0))
    if Z.shape[1]:
        Z = (Z - scaling["z_mean"]) / scaling["z_sd"]
    inp = MMMInputs(
        channels=list(channels), y=df[revenue_col].to_numpy(float) / y_scale, y_scale=y_scale,
        spend_scale=scale, x_lag=lag_matrix(spend_history / scale, L_LAGS), spend=spend,
        spend_history=spend_history, fourier=fourier(np.arange(T)), controls=Z,
        control_names=list(controls), trend=np.arange(T) / scaling["T_ref"], scaling=scaling,
    )
    for lt in lift_tests or []:
        j = channels.index(lt.channel)
        f, m = lt.test_share, lt.multiplier
        off = L_LAGS - 1
        win = slice(off + lt.start, off + lt.start + lt.weeks)
        base = spend_history[:, j].copy()
        # national spend in test weeks is ((1-f) + f m) x; per-capita spend in control geos is x
        base[win] /= (1 - f) + f * m
        test = base.copy()
        test[win] *= m
        rows = slice(lt.start, lt.start + lt.weeks + lt.cooldown)
        inp.lift.append({
            "j": j, "f": f,
            "base": lag_matrix(base / scale[j], L_LAGS)[rows, :, 0],
            "test": lag_matrix(test / scale[j], L_LAGS)[rows, :, 0],
            "obs": lt.lift / y_scale, "se": lt.se / y_scale,
        })
    return inp


def build_model(inp: MMMInputs):
    import pymc as pm
    import pytensor.tensor as pt

    coords = {"channel": inp.channels}
    with pm.Model(coords=coords) as model:
        x_lag = pm.Data("x_lag", inp.x_lag)
        F = pm.Data("fourier", inp.fourier)
        Z = pm.Data("controls", inp.controls)
        tr = pm.Data("trend", inp.trend)
        y = pm.Data("y", inp.y)

        intercept = pm.Normal("intercept", 0.75, 0.25)
        b_trend = pm.Normal("trend_coef", 0.0, 0.2)
        g_f = pm.Normal("fourier_coef", 0.0, 0.1, shape=inp.fourier.shape[1])
        g_z = pm.Normal("control_coef", 0.0, 0.1, shape=inp.controls.shape[1])
        alpha = pm.Beta("alpha", 2.0, 2.0, dims="channel")
        K = pm.LogNormal("K", 0.0, 0.4, dims="channel")
        h = pm.LogNormal("h", np.log(1.3), 0.3, dims="channel")
        beta = pm.HalfNormal("beta", 0.3, dims="channel")
        sigma = pm.HalfNormal("sigma", 0.05)

        lags = pt.arange(L_LAGS)[:, None]
        w = alpha[None, :] ** lags
        w = w / w.sum(axis=0, keepdims=True)

        def hill_t(s):
            s = pt.maximum(s, 1e-9)
            return s**h / (s**h + K**h)

        ad = (x_lag * w[None, :, :]).sum(axis=1)
        media = beta * hill_t(ad)
        mu = intercept + b_trend * tr + pt.dot(F, g_f) + pt.dot(Z, g_z) + media.sum(axis=1)
        pm.Normal("revenue", mu, sigma, observed=y)

        for k, lt in enumerate(inp.lift):
            j = lt["j"]
            base = pm.Data(f"lift{k}_base", lt["base"])
            test = pm.Data(f"lift{k}_test", lt["test"])
            obs = pm.Data(f"lift{k}_obs", np.array(lt["obs"]))
            se = pm.Data(f"lift{k}_se", np.array(lt["se"]))
            wj = w[:, j]
            s_base = pt.maximum((base * wj[None, :]).sum(axis=1), 1e-9)
            s_test = pt.maximum((test * wj[None, :]).sum(axis=1), 1e-9)

            def hj(s, j=j):
                return s ** h[j] / (s ** h[j] + K[j] ** h[j])

            model_lift = lt["f"] * beta[j] * (hj(s_test) - hj(s_base)).sum()
            pm.Normal(f"lift_test_{k}", model_lift, se, observed=obs)
    return model


def _data_dict(inp: MMMInputs) -> dict:
    d = {"x_lag": inp.x_lag, "fourier": inp.fourier, "controls": inp.controls, "trend": inp.trend, "y": inp.y}
    for k, lt in enumerate(inp.lift):
        d.update({f"lift{k}_base": lt["base"], f"lift{k}_test": lt["test"],
                  f"lift{k}_obs": np.array(lt["obs"]), f"lift{k}_se": np.array(lt["se"])})
    return d


@dataclass
class MMMFit:
    inputs: MMMInputs
    draws: dict  # name -> (n_draws, ...) arrays
    diagnostics: dict = field(default_factory=dict)

    @property
    def channels(self) -> list[str]:
        return self.inputs.channels

    def _media_params(self):
        d = self.draws
        return d["alpha"], d["K"], d["h"], d["beta"]

    def contributions(self, spend_history: np.ndarray | None = None) -> np.ndarray:
        """Posterior weekly contributions (draws, T, C) in revenue units."""
        a, K, h, b = self._media_params()
        inp = self.inputs
        x_lag = inp.x_lag if spend_history is None else lag_matrix(spend_history / inp.spend_scale, L_LAGS)
        out = np.empty((len(a), x_lag.shape[0], len(inp.channels)))
        for i in range(len(a)):
            out[i] = b[i] * hill(adstock(x_lag, a[i]), K[i], h[i])
        return out * inp.y_scale

    def roas(self) -> np.ndarray:
        """Posterior ROAS per channel (draws, C) = in-sample contribution / in-sample spend."""
        return self.contributions().sum(axis=1) / self.inputs.spend.sum(axis=0)

    def mroas(self, bump: float = 0.01) -> np.ndarray:
        base = self.contributions()
        out = np.empty((base.shape[0], len(self.channels)))
        for j in range(len(self.channels)):
            sh = self.inputs.spend_history.copy()
            sh[:, j] *= 1 + bump
            up = self.contributions(sh)
            out[:, j] = (up[:, :, j].sum(1) - base[:, :, j].sum(1)) / (bump * self.inputs.spend[:, j].sum())
        return out

    def predict(self, inp: MMMInputs | None = None) -> np.ndarray:
        """Posterior mean revenue (draws, T) for ``inp`` (prepared with this fit's scaling)."""
        inp = inp or self.inputs
        d = self.draws
        a, K, h, b = self._media_params()
        base = (d["intercept"][:, None] + d["trend_coef"][:, None] * inp.trend[None, :]
                + d["fourier_coef"] @ inp.fourier.T + d["control_coef"] @ inp.controls.T)
        media = np.stack([(b[i] * hill(adstock(inp.x_lag, a[i]), K[i], h[i])).sum(1) for i in range(len(a))])
        return (base + media) * self.inputs.y_scale

    def weekly_response(self, weekly_spend: np.ndarray) -> np.ndarray:
        """Steady-state weekly contribution (draws, C) at constant weekly spend per channel."""
        a, K, h, b = self._media_params()
        s = np.asarray(weekly_spend, float) / self.inputs.spend_scale
        return b * hill(s[None, :], K, h) * self.inputs.y_scale

    def summary(self, alpha: float = 0.10) -> pd.DataFrame:
        r, m = self.roas(), self.mroas()
        q = [alpha / 2, 0.5, 1 - alpha / 2]
        rows = []
        for j, c in enumerate(self.channels):
            rq, mq = np.quantile(r[:, j], q), np.quantile(m[:, j], q)
            rows.append({"channel": c, "spend": self.inputs.spend[:, j].sum(),
                         "roas_mean": r[:, j].mean(), "roas_low": rq[0], "roas_median": rq[1], "roas_high": rq[2],
                         "mroas_mean": m[:, j].mean(), "mroas_low": mq[0], "mroas_high": mq[2],
                         "contribution": r[:, j].mean() * self.inputs.spend[:, j].sum(),
                         "p_roas_above_1": float((r[:, j] > 1).mean()),
                         "p_mroas_above_1": float((m[:, j] > 1).mean())})
        return pd.DataFrame(rows)


def fit_mmm(inp: MMMInputs, draws: int = 1000, tune: int = 1000, chains: int = 2, seed: int = 0,
            cache_key: str | None = None, target_accept: float = 0.8) -> MMMFit:
    """Fit with nutpie. ``cache_key`` reuses a compiled model with new data (same shapes)."""
    import arviz as az
    import nutpie

    if cache_key is not None and cache_key in _COMPILED:
        compiled = _COMPILED[cache_key].with_data(**_data_dict(inp))
    else:
        compiled = nutpie.compile_pymc_model(build_model(inp))
        if cache_key is not None:
            _COMPILED[cache_key] = compiled
    trace = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, seed=seed,
                          progress_bar=False, cores=chains, target_accept=target_accept)
    post = trace.posterior
    names = ["alpha", "K", "h", "beta", "intercept", "sigma", "trend_coef", "fourier_coef", "control_coef"]
    d = {k: post[k].values.reshape(-1, *post[k].shape[2:]) for k in names}
    rhats, esss = [], []
    for k in ["alpha", "K", "h", "beta"]:
        arr = np.asarray(post[k].values)  # (chain, draw, C)
        for c in range(arr.shape[-1]):
            rhats.append(float(az.rhat(arr[..., c])))
            esss.append(float(az.ess(arr[..., c])))
    diag = {
        "max_rhat": max(rhats),
        "min_ess": min(esss),
        "divergences": int(trace.sample_stats["diverging"].values.sum()),
        "draws": int(draws * chains),
    }
    return MMMFit(inputs=inp, draws=d, diagnostics=diag)
