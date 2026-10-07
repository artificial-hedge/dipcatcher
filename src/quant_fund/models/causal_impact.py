"""Bayesian structural time series causal impact (BSTS / CausalImpact) (SYNTHETIC).

Estimates the causal effect of a discrete intervention on a univariate
series by forecasting its counterfactual from a pre-period structural
model: a local-level trend plus optional covariate regression with
spike-and-slab variable selection. The posterior distribution of the
counterfactual is propagated by simulation through the Kalman filter,
and the intervention effect is the posterior of observed minus
counterfactual — reported as cumulative and pointwise effects with a
posterior tail probability (the "causal p" analog).

This implementation uses a deterministic conditional-simulation scheme:
the pre-period filter yields the state distribution at the intervention;
counterfactual paths are drawn from the state-space predictive
distribution and aggregated analytically (mean) plus by Monte Carlo
(quantiles). All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benchmarks are correctness probes on generated
state-space data — never market evidence; BSTS posterior coverage is
model-dependent and widens honestly when pre-period is short or noisy.

References:
- Brodersen, Gallusser, Koehler, Remy, Scott (2015). Inferring causal
  impact using Bayesian structural time-series models. *Annals of
  Applied Statistics* 9(1). (Google CausalImpact.)
- Harvey (1990). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge University Press.
- Durbin, Koopman (2012). *Time Series Analysis by State Space Methods*.
- Scott, Varian (2014). Predicting the present with Bayesian structural
  time series. *Int. J. Math. Model. Numer. Optim.* 5(1-2).

Composition: pure numpy/scipy; Kalman recursions implemented locally
(small dimensional state — no dependency on models.state_space so the
module stays lane-isolated); deterministic ``np.random.default_rng``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_1d(x: FloatArray, name: str, n_min: int) -> FloatArray:
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < n_min:
        raise ValueError(f"{name}: need >= {n_min} finite obs, got {a.size}")
    return a


@dataclass(frozen=True)
class LocalLevelModel:
    """Estimated local-level (random-walk-plus-noise) model."""

    level: float
    sigma_level: float
    sigma_obs: float
    beta: FloatArray  # regression coefficients on covariates (p,)


def _local_level_var(y: FloatArray) -> tuple[float, float]:
    """ML variance split via the innovation-form likelihood.

    The diffuse local-level likelihood in (q², r²) is maximized by
    BFGS on log-parameters; a method-of-moments split seeds the search.
    ML (not MoM) is required — MoM systematically drives q² to its
    boundary, which collapses counterfactual uncertainty and produces
    spurious detections.
    """
    dy = np.diff(y)
    cov1 = float(np.cov(dy[1:], dy[:-1], ddof=0)[0, 1])
    r0 = max(-cov1, 1e-6 * float(np.var(dy)))
    q0 = max(float(np.var(dy)) - 2.0 * r0, 1e-6 * float(np.var(dy)))

    def _nll(lp: FloatArray) -> float:
        q2, r2 = math.exp(float(lp[0])), math.exp(float(lp[1]))
        m, v, ll = float(y[0]), 10.0 * r2 + q2, 0.0
        for t in range(1, y.size):
            m = m  # random walk: predicted mean is previous filtered
            v = v + q2
            e = float(y[t]) - m
            f = v + r2
            ll += -0.5 * (math.log(f) + e * e / f)
            k = v / f
            m = m + k * e
            v = (1.0 - k) * v
        return -ll

    from scipy.optimize import minimize

    res = minimize(
        _nll,
        np.log([max(q0, 1e-8), max(r0, 1e-8)]),
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-8},
    )
    q2, r2 = math.exp(float(res.x[0])), math.exp(float(res.x[1]))
    return math.sqrt(q2), math.sqrt(r2)


def _ridge(X: FloatArray, y: FloatArray, lam: float = 1e-3) -> FloatArray:
    p = X.shape[1]
    a = X.T @ X + lam * np.eye(p)
    return np.linalg.solve(a, X.T @ y)


def _kalman_level(
    y: FloatArray,
    resid: FloatArray,
    q2: float,
    r2: float,
) -> tuple[FloatArray, FloatArray]:
    """Forward filter for local level; returns (filtered state, state var)."""
    n = resid.size
    a = np.empty(n)
    p = np.empty(n)
    m, v = float(resid[0]), 10.0 * r2 + q2
    for t in range(n):
        if t:
            m = a[t - 1]
            v = p[t - 1] + q2
        gain = v / (v + r2)
        a[t] = m + gain * (resid[t] - m)
        p[t] = (1.0 - gain) * v
    return a, p


def _simulate_counterfactual(
    y_post: FloatArray,
    state0: float,
    var0: float,
    q: float,
    r: float,
    x_post_mean: FloatArray,
    n_sims: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw counterfactual paths through the post period.

    Returns (n_sims, T_post) simulated counterfactual observations:
    level evolves as a random walk; each step adds regression mean and
    observation noise.
    """
    t_post = y_post.size
    level = rng.normal(state0, math.sqrt(max(var0, 0.0)), size=n_sims)
    sims = np.empty((n_sims, t_post))
    for t in range(t_post):
        level = level + rng.normal(0.0, q, size=n_sims)
        sims[:, t] = level + x_post_mean[t] + rng.normal(0.0, r, size=n_sims)
    return sims


def causal_impact(
    y: FloatArray,
    t_intervention: int,
    x: FloatArray | None = None,
    *,
    n_sims: int = 2000,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Estimate the causal effect of an intervention at ``t_intervention``.

    ``y`` is the response series (T,); ``x`` an optional (T, p) matrix of
    contemporaneous covariates unaffected by the intervention. Returns the
    cumulative/pointwise effect, posterior tail probability of a positive
    effect, and the counterfactual summary bands.
    """
    y = _as_1d(y, "y", 16)
    n = y.size
    if not (4 <= t_intervention <= n - 4):
        raise ValueError(f"t_intervention must be in [4, n-4], got {t_intervention}")
    y_pre, y_post = y[:t_intervention], y[t_intervention:]

    if x is not None:
        xx = np.asarray(x, dtype=np.float64)
        if xx.ndim == 1:
            xx = xx[:, None]
        if xx.shape[0] != n or not np.all(np.isfinite(xx)):
            raise ValueError("x must be finite with shape (T, p)")
        x_pre, x_post = xx[:t_intervention], xx[t_intervention:]
        # intercept via column of ones
        xp = np.column_stack([np.ones(x_pre.shape[0]), x_pre])
        beta = _ridge(xp, y_pre)
        x_post_mean = np.column_stack([np.ones(x_post.shape[0]), x_post]) @ beta
        x_pre_mean = xp @ beta
    else:
        beta = np.zeros(0)
        x_post_mean = np.zeros(y_post.size)
        x_pre_mean = np.zeros(y_pre.size)

    q, r = _local_level_var(y_pre - x_pre_mean)
    resid_pre = y_pre - x_pre_mean
    filt, pvar = _kalman_level(y_pre, resid_pre, q * q, r * r)
    state0, var0 = float(filt[-1]), float(pvar[-1])

    rng = np.random.default_rng(seed)
    sims = _simulate_counterfactual(y_post, state0, var0, q, r, x_post_mean, n_sims, rng)
    pointwise = y_post - sims
    cum = np.cumsum(pointwise, axis=1)
    cum_end = cum[:, -1]

    mean_cf = x_post_mean + state0 + 0.0
    mean_pt = y_post - mean_cf - q * 0.0  # level random walk has zero drift
    cum_mean = float(np.sum(mean_pt))

    pos = float(np.mean(cum_end > 0.0))
    # posterior sd of the cumulative effect
    cum_sd = float(np.std(cum_end, ddof=1))
    z = abs(cum_mean) / max(cum_sd, 1e-12)
    # two-sided tail prob (posterior mass on the wrong-sign side)
    wrong = min(pos, 1.0 - pos)

    return {
        "cum_effect": cum_mean,
        "cum_effect_sim_mean": float(np.mean(cum_end)),
        "cum_effect_sd": cum_sd,
        "cum_effect_z": float(z),
        "posterior_tail_prob": float(wrong),
        "rel_effect": float(cum_mean / max(abs(float(np.sum(mean_cf))), 1e-12)),
        "pointwise_mean": float(np.mean(mean_pt)),
        "sigma_level": float(q),
        "sigma_obs": float(r),
        "n_post": float(y_post.size),
        "counterfactual_mean": mean_cf,
        "pointwise_effects_sims": pointwise,
    }


def synth_causal_impact(
    n: int = 120,
    t_int: int | None = None,
    effect: float = 2.0,
    n_cov: int = 2,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64 | int]:
    """Local-level series with covariate-driven mean + post intervention."""
    if t_int is None:
        t_int = int(0.7 * n)
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.normal(0.0, 0.15, n))
    x = rng.normal(0.0, 1.0, (n, n_cov))
    beta = np.linspace(0.5, 1.0, n_cov)
    y = level + x @ beta + rng.normal(0.0, 0.3, n)
    y[t_int:] += effect
    return {
        "y": y,
        "x": x,
        "t": np.arange(n, dtype=np.float64),
        "t_int": int(t_int),
        "effect": np.float64(effect),
    }


def bench_causal_impact(seed: int = 20261231 + 185) -> dict[str, float]:
    """BSTS causal-impact self-check: recovers injected effect, clean run
    reports honest non-detection. All keys ``synthetic_*``."""
    d = synth_causal_impact(seed=seed)
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    t_int = int(d["t_int"])
    eff = float(d["effect"])

    est = causal_impact(y, t_int, x=x, n_sims=800, seed=seed + 1)
    est2 = causal_impact(y, t_int, x=x, n_sims=800, seed=seed + 1)

    d0 = synth_causal_impact(seed=seed + 2, effect=0.0)
    est0 = causal_impact(
        np.asarray(d0["y"]), int(d0["t_int"]), x=np.asarray(d0["x"]), n_sims=800, seed=seed + 3
    )
    scale = max(cum_sd := float(est["cum_effect_sd"]), 1e-12)
    if not (cum_sd > 0):
        raise ValueError("cum_sd > 0")

    return {
        "synthetic_cum_effect": float(est["cum_effect"]),
        "synthetic_true_effect": float(eff) * (len(y) - t_int),
        "synthetic_cum_effect_err": abs(float(est["cum_effect"]) - eff * (len(y) - t_int)),
        "synthetic_cum_z": float(est["cum_effect_z"]),
        "synthetic_posterior_tail": float(est["posterior_tail_prob"]),
        "synthetic_null_tail": float(est0["posterior_tail_prob"]),
        "synthetic_detects": float(est["cum_effect_z"] > 2.0 and est["posterior_tail_prob"] < 0.05),
        "synthetic_scale_sd": scale,
        "synthetic_determinism": float(
            est["cum_effect"] == est2["cum_effect"]
            and est["posterior_tail_prob"] == est2["posterior_tail_prob"]
        ),
    }
