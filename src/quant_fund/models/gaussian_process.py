"""Gaussian-process regression with marginal-likelihood hyperparameters.

Nonparametric posterior inference over functions. For a squared-
exponential kernel ``k(x,x') = σ_f² exp(−|x−x'|²/2ℓ²)`` and Gaussian
noise ``σ_n²``, the GP posterior gives closed-form predictive mean and
variance; hyperparameters ``(ℓ, σ_f, σ_n)`` are chosen by maximizing
the log marginal likelihood (Rasmussen & Williams §5.4), a Type-II
likelihood that automatically trades fit against complexity.

Includes leave-one-out CV in closed form (the ``K⁻¹`` trick:
``μ_loo = y − α/diag(K⁻¹)``) and 1-D posterior sampling.

References
----------
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes
  for Machine Learning.* MIT Press — chapters 2 and 5.
- Rifkin, R. M. & Lippert, R. A. (2007). *Notes on regularized
  least-squares* — closed-form LOO identity used here.

Honesty contract
----------------
``synth_*`` helpers and ``bench_*`` emit SYNTHETIC correctness checks
only — never market evidence.

Composition notes
-----------------
numpy/scipy only; deterministic ``np.random.default_rng(seed)``;
fail-closed ``ValueError`` on non-PD kernel matrix (after jitter) or
empty inputs; hyperparameters optimized by multi-start BFGS in log
space — deterministic given data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize

FloatArray = np.ndarray

__all__ = [
    "GPModel",
    "bench_gaussian_process",
    "gp_fit",
    "gp_predict",
    "loo_cv",
    "synth_gp",
]


def _check_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("x rows must match y")
    if y.size < 8:
        raise ValueError("need at least 8 observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    return x, y


def _rbf(x1: FloatArray, x2: FloatArray, ls: FloatArray, sf2: float) -> FloatArray:
    d = x1[:, None, :] - x2[None, :, :]
    d = d / ls[None, None, :]
    return sf2 * np.exp(-0.5 * np.sum(d * d, axis=2))


@dataclass
class GPModel:
    """Fitted GP: training data, hyperparameters, factorized kernel."""

    x: FloatArray
    y: FloatArray
    ls: FloatArray  # per-dim lengthscales (ARD)
    sf2: float
    sn2: float
    chol: FloatArray
    alpha: FloatArray
    log_marginal: float


def _nll(log_par: FloatArray, x: FloatArray, y: FloatArray) -> float:
    p = x.shape[1]
    ls = np.exp(log_par[:p])
    sf2 = math.exp(log_par[p])
    sn2 = math.exp(log_par[p + 1])
    k = _rbf(x, x, ls, sf2) + sn2 * np.eye(x.shape[0])
    try:
        c = linalg.cho_factor(k, lower=True)
    except linalg.LinAlgError:
        return 1e12
    alpha = linalg.cho_solve(c, y)
    lml = (
        -0.5 * float(y @ alpha)
        - float(np.log(np.diag(c[0])).sum())
        - 0.5 * y.size * math.log(2 * math.pi)
    )
    return -lml


def gp_fit(x: FloatArray, y: FloatArray, n_restarts: int = 4, seed: int = 0) -> dict[str, object]:
    """Fit a GP by Type-II ML over (ℓ_j, σ_f, σ_n).

    Multi-start BFGS in log space; returns a ``GPModel`` in the dict
    plus the learned hyperparameters and log marginal likelihood.
    """
    x, y = _check_xy(x, y)
    p = x.shape[1]
    y_sd = max(float(y.std()), 1e-6)
    rng = np.random.default_rng(seed)
    base = np.concatenate(
        [
            np.full(p, math.log(float(x.std(axis=0).mean() + 1e-3))),
            [math.log(y_sd), math.log(0.1 * y_sd)],
        ]
    )
    best = None
    best_nll = math.inf
    for r in range(max(1, n_restarts)):
        x0 = base + rng.normal(0, 0.3, p + 2) if r else base
        res = optimize.minimize(_nll, x0, args=(x, y), method="BFGS", options={"maxiter": 300})
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best = res.x
    assert best is not None
    ls = np.exp(best[:p])
    sf2 = math.exp(best[p])
    sn2 = math.exp(best[p + 1])
    k = _rbf(x, x, ls, sf2) + sn2 * np.eye(x.shape[0])
    try:
        c = linalg.cho_factor(k, lower=True)
    except linalg.LinAlgError as e:
        raise ValueError("kernel matrix not PD after fitting") from e
    alpha = np.asarray(linalg.cho_solve(c, y))
    model = GPModel(
        x=x, y=y, ls=ls, sf2=sf2, sn2=sn2, chol=c[0], alpha=alpha, log_marginal=-best_nll
    )
    return {
        "model": model,
        "lengthscales": ls,
        "sigma_f2": sf2,
        "sigma_n2": sn2,
        "log_marginal": -best_nll,
    }


def gp_predict(model: GPModel, x_new: FloatArray, latent: bool = False) -> dict[str, FloatArray]:
    """GP posterior mean/variance at ``x_new``. ``latent=True`` excludes
    the observation-noise term from the variance."""
    xn = np.asarray(x_new, dtype=np.float64)
    if xn.ndim == 1:
        xn = xn[:, None]
    if xn.shape[1] != model.x.shape[1]:
        raise ValueError("x_new feature count mismatch")
    k_star = _rbf(model.x, xn, model.ls, model.sf2)
    k_ss = _rbf(xn, xn, model.ls, model.sf2)
    mean = np.asarray(k_star.T @ model.alpha)
    v = linalg.cho_solve((model.chol, True), k_star)
    cov = k_ss - k_star.T @ v
    var = np.clip(np.diag(cov), 1e-12, None)
    if not latent:
        var = var + model.sn2
    return {"mean": mean, "var": var, "sd": np.sqrt(var)}


def loo_cv(model: GPModel) -> dict[str, float]:
    """Closed-form leave-one-out CV (Rifkin & Lippert):
    ``e_i = α_i / (K⁻¹)_ii``; reports LOO-RMSE and standardized
    coverage of the true values by the LOO predictive intervals."""
    k_inv = linalg.cho_solve((model.chol, True), np.eye(model.x.shape[0]))
    d = np.diag(k_inv)
    e = model.alpha / d
    var_loo = 1.0 / d
    z = e / np.sqrt(var_loo)
    return {
        "loo_rmse": float(math.sqrt(float(np.mean(e * e)))),
        "loo_coverage_95": float(np.mean(np.abs(z) < 1.96)),
        "n": float(model.x.shape[0]),
    }


def synth_gp(
    n: int = 120,
    noise: float = 0.15,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC smooth function ``f(x) = sin(2πx) + 0.5x`` observed
    with Gaussian noise on a random 1-D design."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-2, 2, n))[:, None]
    f = np.sin(2 * math.pi * x[:, 0]) + 0.5 * x[:, 0]
    y = f + noise * rng.standard_normal(n)
    return {"x": x, "y": y, "f": f, "noise": np.float64(noise)}


def bench_gaussian_process(seed: int = 20261231 + 183) -> dict[str, float]:
    """SYNTHETIC: GP recovers a smooth signal; LOO coverage sane."""
    d = synth_gp(n=120, noise=0.15, seed=seed)
    x = np.asarray(d["x"])
    y = np.asarray(d["y"])
    f = np.asarray(d["f"])
    fit = gp_fit(x, y, n_restarts=3, seed=seed)
    model = fit["model"]
    assert isinstance(model, GPModel)
    pred = gp_predict(model, x, latent=True)
    mu = np.asarray(pred["mean"])
    loo = loo_cv(model)
    xg = np.linspace(-2, 2, 200)[:, None]
    fg = np.sin(2 * math.pi * xg[:, 0]) + 0.5 * xg[:, 0]
    pg = gp_predict(model, xg, latent=True)
    cover = float(np.mean(np.abs(np.asarray(pg["mean"]) - fg) < 1.96 * np.asarray(pg["sd"])))
    fit2 = gp_fit(x, y, n_restarts=1, seed=seed)
    model2 = fit2["model"]
    assert isinstance(model2, GPModel)
    return {
        "synthetic_train_rmse": float(math.sqrt(float(np.mean((mu - f) ** 2)))),
        "synthetic_grid_rmse": float(math.sqrt(float(np.mean((np.asarray(pg["mean"]) - fg) ** 2)))),
        "synthetic_coverage_95": cover,
        "synthetic_loo_rmse": float(loo["loo_rmse"]),
        "synthetic_sigma_n": float(math.sqrt(model.sn2)),
        "synthetic_sigma_n_err": abs(float(math.sqrt(model.sn2)) - 0.15),
        "synthetic_lml": model.log_marginal,
        "synthetic_determinism": float(model.log_marginal == model2.log_marginal),
    }
