"""Saddlepoint approximations — Daniels (1954), Lugannani & Rice (1980).

For i.i.d. sums S_n = sum X_i with cumulant-generating function K(t),
the saddlepoint density of the mean X_bar at x is

    f_hat(x) = sqrt(n / (2 pi K''(t_hat))) * exp(n (K(t_hat) - t_hat x))

where t_hat solves K'(t_hat) = x. The Lugannani-Rice tail
approximation is

    P(X_bar > x) ~= 1 - Phi(w) + phi(w) (1/u - 1/w),

    w = sign(t_hat) sqrt(2 n (t_hat x - K(t_hat))),
    u = t_hat sqrt(n K''(t_hat)).

These give far better small-sample tail accuracy than the central
limit Gaussian; the bench verifies against the exact Gamma
distribution (where the true CDF of the mean is known in closed
form) and against a Gaussian mean (where the approximation must be
exact up to the normalization constant).

References
----------
- Daniels (1954) Ann. Math. Stat., "Saddlepoint approximations in
  statistics".
- Lugannani & Rice (1980) Adv. Appl. Prob. 12, "Saddle point
  approximation for the distribution of the sum of independent
  random variables".

Honesty
-------
Deterministic numerical math (Newton on K' = x, renormalized on the
density grid). The bench reports the worst CDF error vs exact
distributions — SYNTHETIC, no market claims.

Composition
-----------
Called by ``quant_fund.research.benches_w64.bench_saddlepoint``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]

_TOL = 1e-12


def _solve_saddle(k1: float, k2: float, x: float, n: int, kind: str) -> float:
    """Solve K'(t) = x by Newton on the supported family."""
    t = 0.0
    for _ in range(60):
        if kind == "gamma":
            f = k1 / (k2 - t) - x
            fp = k1 / (k2 - t) ** 2
            bound = k2 * 0.999999
        else:  # normal
            f = k1 + k2 * t - x
            fp = k2
            bound = np.inf
        if fp <= 0:
            raise ValueError("Newton step non-positive curvature")
        t_new = t - f / fp
        if t_new >= bound:
            t_new = (t + bound) / 2.0
        if abs(t_new - t) < _TOL:
            return t_new
        t = t_new
    return t


def _cgh(
    kind: str, k1: float, k2: float, t: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(K(t), K'(t), K''(t)) for the supported families.

    gamma:  X_i ~ Gamma(shape=k1, rate=k2), K(t) = -k1 ln(1 - t/k2).
    normal: X_i ~ N(k1, k2),           K(t) = k1 t + k2 t^2/2.
    """
    if kind == "gamma":
        z = 1.0 - t / k2
        if np.any(z <= 0):
            raise ValueError("t outside CGF domain")
        return -k1 * np.log(z), k1 / (k2 * z), k1 / (k2 * z) ** 2
    if kind == "normal":
        return k1 * t + k2 * t**2 / 2.0, k1 + k2 * t, np.full(np.shape(t), k2)
    raise ValueError("unsupported family")


def saddlepoint_pdf(
    x: FloatArray, kind: str, k1: float, k2: float, n: int, renormalize: bool = True
) -> FloatArray:
    """Renormalized saddlepoint density of the sample mean.

    Fail-closed on non-positive n, bad family, or non-finite grid.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite 1-D grid")
    if n < 1:
        raise ValueError("n must be >= 1")
    t = np.array([_solve_saddle(k1, k2, xi, n, kind) for xi in x])
    kk, _, k2v = _cgh(kind, k1, k2, t)
    out = np.sqrt(n / (2.0 * np.pi * k2v)) * np.exp(n * (kk - t * x))
    if renormalize:
        dx = np.gradient(x)
        mass = float(np.sum(out * dx))
        if mass > 0:
            out = out / mass
    return np.asarray(out, dtype=np.float64)


def lugannani_rice(x: FloatArray, kind: str, k1: float, k2: float, n: int) -> FloatArray:
    """Upper-tail probability P(X_bar > x) via Lugannani-Rice."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    t = np.array([_solve_saddle(k1, k2, xi, n, kind) for xi in x])
    kk, _, k2v = _cgh(kind, k1, k2, t)
    w = np.sign(t) * np.sqrt(np.maximum(2.0 * n * (t * x - kk), 0.0))
    u = t * np.sqrt(n * k2v)
    inv_u = np.where(np.abs(u) > _TOL, 1.0 / np.where(np.abs(u) > _TOL, u, 1.0), 0.0)
    inv_w = np.where(np.abs(w) > _TOL, 1.0 / np.where(np.abs(w) > _TOL, w, 1.0), 0.0)
    tail = 1.0 - norm.cdf(w) + norm.pdf(w) * (inv_u - inv_w)
    tail = np.where(np.abs(t) < 1e-7, 1.0 - norm.cdf(0.0) * 0.0 + 1.0 - norm.cdf(w), tail)
    return np.asarray(np.clip(tail, 0.0, 1.0), dtype=np.float64)


def bench_saddlepoint(seed: int = 20261231 + 372) -> dict[str, float]:
    """SYNTHETIC check — LR tail error vs exact Gamma + normal exactness."""
    _ = np.random.default_rng(seed)  # deterministic bench; seed stamps the family
    # Exact Gamma mean: X_bar ~ Gamma(n*k1, rate=k2*n) -> compare upper tails.
    from scipy.stats import gamma as sg

    n, shape, rate = 8, 3.0, 2.0
    grid = np.linspace(shape / rate * 0.4, shape / rate * 2.6, 40)
    lr = lugannani_rice(grid, "gamma", shape, rate, n)
    exact = 1.0 - sg.cdf(grid * n, a=n * shape, scale=1.0 / rate)
    err_g = float(np.max(np.abs(lr - exact)))
    # Normal family: LR reduces to the Gaussian tail (u == w modulo
    # the correction term -> near-exact for the mean of normals).
    lr_n = lugannani_rice(np.array([0.5, 1.0, 1.5]), "normal", 0.0, 1.0, 16)
    ex_n = 1.0 - norm.cdf(np.array([0.5, 1.0, 1.5]) * np.sqrt(16.0))
    err_n = float(np.max(np.abs(lr_n - ex_n)))
    # Density shape: renormalized saddlepoint pdf close to exact Gamma pdf.
    pdf = saddlepoint_pdf(grid, "gamma", shape, rate, n)
    ex_pdf = sg.pdf(grid * n, a=n * shape, scale=1.0 / rate) * n
    dens_err = float(np.max(np.abs(pdf - ex_pdf) / np.max(ex_pdf)))
    if err_g > 0.02:
        raise ValueError("Lugannani-Rice tail error too large on Gamma")
    if err_n > 0.02:
        raise ValueError("Lugannani-Rice not near-exact on normal")
    if dens_err > 0.06:
        raise ValueError("saddlepoint density deviates from exact")
    return {
        "synthetic_sp_err_gamma": err_g,
        "synthetic_sp_err_normal": err_n,
        "synthetic_sp_dens_err": dens_err,
        "synthetic_score": 1.0,
    }
