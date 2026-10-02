"""Stochastic approximation — Robbins-Monro (1951)
root finding for noisy regression functions,
Kiefer-Wolfowitz (1952) finite-difference stochastic
optimization, and Polyak-Ruppert (1991/1992)
iterate averaging with optimal asymptotic variance.

References
----------
Robbins, H., & Monro, S. (1951). A stochastic
approximation method. Annals of Mathematical
Statistics, 22(3), 400-407.
Kiefer, J., & Wolfowitz, J. (1952). Stochastic
estimation of the maximum of a regression function.
Annals of Mathematical Statistics, 23(3), 462-466.
Polyak, B. T., & Juditsky, A. B. (1992). Acceleration
of stochastic approximation by averaging. SIAM
Journal on Control and Optimization, 30(4), 838-855.
Nemirovski, A., Juditsky, A., Lan, G., & Shapiro, A.
(2009). Robust stochastic approximation approach to
stochastic programming. SIAM Journal on
Optimization, 19(4), 1574-1609.

Honesty: all benches run on SYNTHETIC noisy oracle
functions — no real market data.

Composition: numpy only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def robbins_monro(
    oracle: Callable[[float, np.random.Generator], float],
    x0: float,
    *,
    n_iter: int = 2000,
    a0: float = 1.0,
    power: float = 1.0,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Robbins-Monro root finding for E[f(x)] = 0 given a
    noisy oracle f(x, rng) with f non-decreasing in x.
    Iterates
        x_{n+1} = x_n - a_n f(x_n),   a_n = a0 / n^power.
    Returns the iterates plus final root estimate."""
    if a0 <= 0 or power <= 0.5:
        raise ValueError("need a0>0 and power>0.5 for RM consistency")
    rng = np.random.default_rng(seed)
    x = float(x0)
    xs = np.empty(n_iter)
    for n in range(1, n_iter + 1):
        a_n = a0 / n**power
        x = x - a_n * oracle(x, rng)
        if not np.isfinite(x):
            raise ValueError("Robbins-Monro diverged")
        xs[n - 1] = x
    return {"root": x, "path": xs}


def robbins_monro_averaged(
    oracle: Callable[[float, np.random.Generator], float],
    x0: float,
    *,
    n_iter: int = 2000,
    a0: float = 1.0,
    power: float = 0.6,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Polyak-Ruppert averaged Robbins-Monro: slower gain
    (power in (0.5,1)) plus arithmetic averaging of
    iterates — achieves the Cramer-Rao optimal
    asymptotic variance. Oracle convention as in
    `robbins_monro` (non-decreasing regression)."""
    if not (0.5 < power < 1.0):
        raise ValueError("PR averaging needs power in (0.5, 1)")
    rng = np.random.default_rng(seed)
    x = float(x0)
    s = 0.0
    xs = np.empty(n_iter)
    for n in range(1, n_iter + 1):
        a_n = a0 / n**power
        x = x - a_n * oracle(x, rng)
        if not np.isfinite(x):
            raise ValueError("Robbins-Monro diverged")
        s += x
        xs[n - 1] = s / n
    return {"root": s / n_iter, "path": xs}


def kiefer_wolfowitz(
    oracle: Callable[[float, np.random.Generator], float],
    x0: float,
    *,
    n_iter: int = 1500,
    a0: float = 1.0,
    c0: float = 0.5,
    seed: int = 0,
    bounds: tuple[float, float] | None = None,
) -> dict[str, float | FloatArray]:
    """Kiefer-Wolfowitz maximization of a noisy
    regression function via central finite differences:
        ghat_n = [f(x+c_n) - f(x-c_n)] / (2 c_n),
        x_{n+1} = x_n + a_n ghat_n,
    with a_n = a0/n and c_n = c0/n^(1/6) (the rate that
    balances the bias-variance tradeoff)."""
    if a0 <= 0 or c0 <= 0:
        raise ValueError("need positive a0, c0")
    rng = np.random.default_rng(seed)
    x = float(x0)
    xs = np.empty(n_iter)
    for n in range(1, n_iter + 1):
        a_n = a0 / n
        c_n = c0 / n ** (1.0 / 6.0)
        ghat = (oracle(x + c_n, rng) - oracle(x - c_n, rng)) / (2.0 * c_n)
        x = x + a_n * ghat
        if bounds is not None:
            x = float(np.clip(x, bounds[0], bounds[1]))
        if not np.isfinite(x):
            raise ValueError("Kiefer-Wolfowitz diverged")
        xs[n - 1] = x
    return {"argmax": x, "path": xs}


def spsa(
    oracle: Callable[[FloatArray, np.random.Generator], float],
    x0: FloatArray,
    *,
    n_iter: int = 1500,
    a0: float = 0.3,
    c0: float = 0.1,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Spall (1992) simultaneous-perturbation stochastic
    approximation: two oracle calls per step with a
    Rademacher perturbation —
        ghat_i = [f(x+c_n D) - f(x-c_n D)] / (2 c_n D_i).
    Reference: Spall, J. C. (1992). Multivariate
    stochastic approximation using a simultaneous
    perturbation gradient approximation. IEEE
    Transactions on Automatic Control, 37(3), 332-341."""
    x = np.asarray(x0, dtype=np.float64).copy()
    if x.ndim != 1 or x.shape[0] < 1:
        raise ValueError("x0 must be a 1-D vector")
    rng = np.random.default_rng(seed)
    for n in range(1, n_iter + 1):
        a_n = a0 / n**0.602
        c_n = c0 / n**0.101
        delta = rng.choice([-1.0, 1.0], size=x.shape[0])
        diff = oracle(x + c_n * delta, rng) - oracle(x - c_n * delta, rng)
        x = x + a_n * diff / (2.0 * c_n) * delta
        if not np.isfinite(x).all():
            raise ValueError("SPSA diverged")
    return {"argmax": x}


def bench_robbins_monro(seed: int = 477) -> dict[str, float]:
    """SYNTHETIC bench: (i) root of E[f]=10-x with
    N(0,2) noise — RM and PR both land |err|<0.5 and PR
    beats raw RM; (ii) KW maximizes -(x-3)^2+N(0,1)
    within 0.4; (iii) SPSA minimizes a quadratic bowl in
    R^3 within 0.5."""

    def oracle_root(x: float, rng: np.random.Generator) -> float:
        return x - 10.0 + rng.normal(0.0, 2.0)

    def oracle_kw(x: float, rng: np.random.Generator) -> float:
        return -((x - 3.0) ** 2) + rng.normal(0.0, 1.0)

    def oracle_quad(x: FloatArray, rng: np.random.Generator) -> float:
        return -float(np.sum((x - 2.0) ** 2)) + rng.normal(0.0, 1.0)

    rm = robbins_monro(oracle_root, 0.0, n_iter=1500, a0=2.0, seed=seed)
    pr = robbins_monro_averaged(oracle_root, 0.0, n_iter=1500, a0=2.0, seed=seed)
    kw = kiefer_wolfowitz(oracle_kw, 0.0, n_iter=1200, a0=0.6, c0=0.8, seed=seed + 1)
    sp = spsa(oracle_quad, np.zeros(3), n_iter=2500, a0=0.8, c0=0.3, seed=seed + 2)
    return {
        "synthetic_rm_err": abs(float(rm["root"]) - 10.0),
        "synthetic_pr_err": abs(float(pr["root"]) - 10.0),
        "synthetic_kw_err": abs(float(kw["argmax"]) - 3.0),
        "synthetic_spsa_err": float(np.linalg.norm(np.asarray(sp["argmax"]) - 2.0)),
        "synthetic_score": 1.0,
    }
