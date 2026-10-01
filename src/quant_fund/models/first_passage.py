"""First-passage times for drifted Brownian motion.

Hitting time of the level ``b`` by X_t = x0 + v t + sigma W_t follows
an inverse-Gaussian law IG(mu, lambda) with

    mu     = (b - x0) / v          (v > 0, b > x0)
    lambda = ((b - x0) / sigma)^2

and CDF

    F(t) = Phi(sqrt(lam/t)(t/mu - 1))
           + e^{2 lam/mu} Phi(-sqrt(lam/t)(t/mu + 1)).

For discretely monitored barriers (observation grid dt), Siegmund's
corrected-continuity approximation treats the continuous barrier as
``b + beta * sigma * sqrt(dt)`` with beta = zeta(1/2)/sqrt(2 pi)
= 0.5826 — the discrete-hit probability at grid resolution equals the
continuous-hit probability to the lifted barrier.

References
----------
- Siegmund (1985) "Sequential Analysis" — corrected-continuity.
- Chhikara & Folks (1989) "The Inverse Gaussian Distribution".

Honesty
-------
The bench validates the analytic CDF against path-simulated hitting
fractions and the Siegmund lift against discretely monitored sims.
SYNTHETIC only — no market claims.

Composition
-----------
Called by ``quant_fund.research.benches_w63.bench_first_passage``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]

_ZETA_HALF = 1.4603545088095868  # Riemann zeta(1/2) magnitude used by Siegmund
_BETA_SIEGMUND = abs(-1.4603545088095868) / float(np.sqrt(2.0 * np.pi))


def _ig_params(x0: float, v: float, sigma: float, b: float) -> tuple[float, float]:
    if not (sigma > 0 and v > 0 and b > x0):
        raise ValueError("require sigma>0, v>0, b>x0")
    d = b - x0
    return d / v, (d / sigma) ** 2


def fp_cdf(t: FloatArray, x0: float, v: float, sigma: float, b: float) -> FloatArray:
    """Inverse-Gaussian hitting-time CDF."""
    t = np.asarray(t, dtype=float)
    if np.any(t <= 0) or not np.all(np.isfinite(t)):
        raise ValueError("t must be positive and finite")
    mu, lam = _ig_params(x0, v, sigma, b)
    z1 = np.sqrt(lam / t) * (t / mu - 1.0)
    z2 = -np.sqrt(lam / t) * (t / mu + 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        out = norm.cdf(z1) + np.exp(2.0 * lam / mu) * norm.cdf(z2)
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float64)


def fp_mean(x0: float, v: float, sigma: float, b: float) -> float:
    """E[T] = (b - x0)/v."""
    mu, _ = _ig_params(x0, v, sigma, b)
    return mu


def siegmund_lift(sigma: float, dt: float) -> float:
    """Corrected-continuity barrier lift: beta * sigma * sqrt(dt)."""
    if not (sigma > 0 and dt > 0):
        raise ValueError("sigma and dt must be positive")
    return _BETA_SIEGMUND * sigma * float(np.sqrt(dt))


def fp_mc(
    x0: float,
    v: float,
    sigma: float,
    b: float,
    t: float,
    n_paths: int,
    seed: int,
    dt: float | None = None,
) -> tuple[float, float]:
    """Simulated hitting probability by time ``t``.

    ``dt=None`` samples continuous approximations on a fine grid;
    passing a coarser ``dt`` emulates discrete monitoring. Returns
    (hit fraction, Monte-Carlo standard error).
    """
    if t <= 0:
        raise ValueError("t must be positive")
    rng = np.random.default_rng(seed)
    n_steps = int(round(t / (dt if dt else 1.0 / 240.0)))
    n_steps = max(2, n_steps)
    step = t / n_steps
    x = np.full(n_paths, x0)
    hit = np.zeros(n_paths, dtype=bool)
    for _ in range(n_steps):
        x = x + v * step + sigma * np.sqrt(step) * rng.standard_normal(n_paths)
        hit |= x >= b
    p = float(np.mean(hit))
    se = float(np.sqrt(p * (1.0 - p) / n_paths))
    return p, se


def bench_first_passage(seed: int = 20261231 + 371) -> dict[str, float]:
    """SYNTHETIC check — IG CDF vs MC, and Siegmund correction."""
    x0, v, sigma, b = 0.0, 0.8, 0.5, 1.0
    t_grid = np.array([0.5, 1.0, 1.5, 2.0])
    cdf = fp_cdf(t_grid, x0, v, sigma, b)
    # MC on a fine grid ≈ continuous.
    mc = np.array([fp_mc(x0, v, sigma, b, t, 30000, seed + i) for i, t in enumerate(t_grid)])
    err = np.abs(cdf - mc[:, 0])
    if float(np.max(err)) > 4.0 * float(np.max(mc[:, 1])) + 0.01:
        raise ValueError("IG CDF disagrees with path simulation")
    if not np.all(np.diff(cdf) > 0):
        raise ValueError("CDF not increasing")
    # Discrete monitoring (dt=0.1) undercounts vs continuous; the
    # Siegmund lift should recover the discrete-hit probability.
    dt = 0.1
    p_disc, se_disc = fp_mc(x0, v, sigma, b, 1.5, 30000, seed + 9, dt=dt)
    lift = siegmund_lift(sigma, dt)
    p_corr = float(fp_cdf(np.array([1.5]), x0, v, sigma, b + lift)[0])
    if p_disc >= cdf[2] + 1e-9:
        raise ValueError("discrete monitoring should undercount hits")
    if abs(p_corr - p_disc) > 4.0 * se_disc + 0.02:
        raise ValueError("Siegmund lift does not recover discrete prob")
    return {
        "synthetic_fp_cdf_2": float(cdf[-1]),
        "synthetic_fp_mc_2": float(mc[-1, 0]),
        "synthetic_fp_max_err": float(np.max(err)),
        "synthetic_fp_disc": p_disc,
        "synthetic_fp_siegmund": p_corr,
        "synthetic_fp_lift": lift,
        "score": 1.0,
    }
