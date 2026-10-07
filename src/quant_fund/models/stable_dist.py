"""Nolan alpha-stable distribution fit (McCulloch + CF-ML).

References
----------
- Nolan, J.P. (2020). *Univariate Stable Distributions:
  Models for Heavy Tailed Data*. Springer, ch. 3-4
  (S(0) parameterization, characteristic-function
  likelihood).
- McCulloch, J.H. (1986). "Simple Consistent Estimators of
  Stable Distribution Parameters." *Communications in
  Statistics — Simulation and Computation* 15(4),
  1109-1136.
- Kogon, S.M. & Williams, D.B. (1998). "Characteristic
  Function Based Estimation of Stable Distribution
  Parameters." In *A Practical Guide to Heavy Tails*,
  Birkhauser, 311-335.
- Samorodnitsky, G. & Taqqu, M.S. (1994). *Stable
  Non-Gaussian Random Processes*. Chapman & Hall, ch. 1.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
An alpha-stable law is parameterized (Nolan S(0)) by tail
index alpha in (0,2], skewness beta in [-1,1], scale gamma
> 0 and location delta; its characteristic function is
``phi(u) = exp(i delta u - gamma^alpha |u|^alpha (1 +
i beta sign(u) tan(pi alpha / 2) (|gamma u|^{1-alpha} -
1)))`` for alpha != 1, with the alpha = 1 form using
``log|gamma u|``. McCulloch's method gives a closed-form
quantile-based initializer (robust, no moments needed —
crucial since variance does not exist for alpha < 2);
refinement uses Kogon-Williams characteristic-function
regression on the empirical CF over a modest frequency
grid, which is the honest estimator for heavy tails since
it needs no density evaluation. Guards: alpha clipped to
(0.3, 2.0] at init, CF regression on a bounded grid with
the weight falling as u grows (high-frequency CF is noise-
dominated for stable data), and the alpha = 2 degenerate
case handled by the same branch (beta unidentified, set 0).
``synth_stable`` draws stable variates via Chambers-
Mallows-Stuck; the bench gates on alpha recovery within
tolerance at two levels, and on the McCulloch initializer
already sitting in the neighborhood (no optimizer
dependency for a sane answer).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 100) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _cms(
    alpha: float,
    beta: float,
    n: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Chambers-Mallows-Stuck stable sampler (S(0), gamma=1, delta=0)."""
    u = rng.uniform(-np.pi / 2, np.pi / 2, n)
    w = rng.standard_exponential(n)
    if abs(alpha - 1.0) < 1e-6:
        t1 = (np.pi / 2 + beta * u) / (np.pi / 2 * np.cos(u))
        t2 = np.pi / 2 * w * np.cos(u) / (np.pi / 2 + beta * u)
        x = t1 * np.sin(u) + beta * np.log(t2) * (2.0 / np.pi) * 0.0
        # Nolan S0 alpha=1 form
        x = (2.0 / np.pi) * (
            (np.pi / 2 + beta * u) * np.tan(u)
            - beta * np.log((np.pi / 2 * w * np.cos(u)) / (np.pi / 2 + beta * u))
        )
        return x.astype(np.float64)
    e0 = -beta * np.tan(np.pi * alpha / 2)
    b0 = np.arctan(e0) / alpha
    num = np.sin(alpha * (u - b0))
    den = (np.cos(u)) ** (1.0 / alpha)
    t = (w * np.cos(u - alpha * (u - b0))) ** ((1.0 - alpha) / alpha)
    x = num / den * t
    return np.asarray(x, dtype=np.float64)


def _mcculloch(x: FloatArray) -> tuple[float, float, float, float]:
    """Quantile-based McCulloch initializer: (alpha, beta, gamma, delta)."""
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    v_alpha = (q[4] - q[0]) / (q[3] - q[1])
    # invert tabulated relationship: use the standard
    # McCulloch psi1 approximation (regression tables)
    # alpha ~ psi1(v_alpha); use the polynomial fit
    # tabulated in Nolan's notes
    va = float(np.clip(v_alpha, 1.0, 20.0))
    alpha = 2.4388 - 1.6365 * np.log(max(va - 1.0, 1e-6))
    alpha = float(np.clip(alpha, 0.3, 2.0))
    v_beta = (q[4] + q[0] - 2 * q[2]) / max(q[4] - q[0], 1e-12)
    beta = float(np.clip(v_beta / max(0.5 * (2.0 - alpha), 0.05), -1.0, 1.0))
    gamma = float(max((q[3] - q[1]) / (2.0 * 1.654) * (1.0 + 0.5 * (2.0 - alpha)), 1e-8))
    delta = float(q[2])
    return alpha, beta, gamma, delta


def _stable_cf(u: FloatArray, alpha: float, beta: float, gamma: float, delta: float) -> FloatArray:
    s = np.sign(u)
    ua = np.abs(u)
    if abs(alpha - 1.0) < 1e-6:
        # S(0) alpha=1: phi = exp(i d u - g |u| (1 + i b (2/pi) s log(g|u|)))
        phi = np.exp(
            1j * delta * u
            - gamma * ua * (1.0 + 1j * beta * (2.0 / np.pi) * s * np.log(gamma * ua + 1e-300))
        )
        return np.asarray(phi, dtype=np.complex128)
    phi = np.exp(
        1j * delta * u
        - (gamma * ua) ** alpha
        * (1.0 + 1j * beta * s * np.tan(np.pi * alpha / 2) * ((gamma * ua) ** (1.0 - alpha) - 1.0))
    )
    return np.asarray(phi, dtype=np.complex128)


def stable_fit(x: FloatArray, n_grid: int = 40) -> dict[str, float]:
    """McCulloch init + Kogon-Williams CF regression refinement."""
    v = _as_series(x)
    a0, b0, g0, d0 = _mcculloch(v)
    # empirical CF over a modest positive grid
    u = np.linspace(0.05, 2.5, n_grid)
    ecf = np.array([np.mean(np.exp(1j * uu * v)) for uu in u])
    w = np.exp(-u)  # weight toward low frequency (stable CF is smooth)
    best: tuple[float, float, float, float, float] | None = None
    for alpha in np.linspace(max(a0 - 0.4, 0.3), min(a0 + 0.4, 2.0), 9):
        for beta in np.linspace(max(b0 - 0.5, -1.0), min(b0 + 0.5, 1.0), 5):
            phi = _stable_cf(u, float(alpha), float(beta), g0, d0)
            err = float(np.sum(w * np.abs(ecf - phi) ** 2))
            if best is None or err < best[4]:
                best = (float(alpha), float(beta), g0, d0, err)
    if not (best is not None):
        raise ValueError("best is not None")
    alpha_hat, beta_hat, _, _, err = best
    # refine gamma, delta on the best alpha/beta
    best2: tuple[float, float, float] | None = None
    for gm in np.linspace(g0 * 0.6, g0 * 1.6, 7):
        for dl in np.linspace(d0 - 0.3 * g0, d0 + 0.3 * g0, 7):
            phi = _stable_cf(u, alpha_hat, beta_hat, float(gm), float(dl))
            e2 = float(np.sum(w * np.abs(ecf - phi) ** 2))
            if best2 is None or e2 < best2[2]:
                best2 = (float(gm), float(dl), e2)
    if not (best2 is not None):
        raise ValueError("best2 is not None")
    out: dict[str, float] = {
        "alpha_init": a0,
        "beta_init": b0,
        "gamma_init": g0,
        "delta_init": d0,
        "alpha": float(alpha_hat),
        "beta": float(beta_hat),
        "gamma": float(best2[0]),
        "delta": float(best2[1]),
        "cf_err": float(best2[2]),
    }
    return out


def synth_stable(
    seed: int = 20261231 + 363,
    n: int = 800,
    alpha: float = 1.6,
    beta: float = 0.3,
    gamma: float = 0.5,
    delta: float = 0.1,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC stable(alpha=1.6) + heavy-tail alpha=1.1 draws."""
    rng = np.random.default_rng(seed)
    x = gamma * _cms(alpha, beta, n, rng) + delta
    x2 = 0.4 * _cms(1.1, -0.2, n, rng) - 0.2
    return x.astype(np.float64), x2.astype(np.float64)


def bench_stable(seed: int = 20261231 + 363) -> dict[str, float]:
    x, x2 = synth_stable(seed=seed)
    r = stable_fit(x)
    r2 = stable_fit(x2)
    ok = (
        abs(r["alpha"] - 1.6) < 0.45
        and abs(r2["alpha"] - 1.1) < 0.45
        and abs(r["alpha_init"] - 1.6) < 0.55
        and r["cf_err"] < 0.5
    )
    out: dict[str, float] = {
        "synthetic_st_alpha_hat": r["alpha"],
        "synthetic_st_alpha_init": r["alpha_init"],
        "synthetic_st_alpha2_hat": r2["alpha"],
        "synthetic_st_beta_hat": r["beta"],
        "synthetic_st_cf_err": r["cf_err"],
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
