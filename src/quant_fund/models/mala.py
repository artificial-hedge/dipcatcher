"""Metropolis-adjusted and unadjusted Langevin Monte Carlo.

Roberts & Tweedie (1996): the Langevin diffusion

    dX_t = (1/2) grad log pi(X_t) dt + dW_t

has stationary density pi. Euler discretization gives the proposal

    x' = x + (h^2 / 2) grad log pi(x) + h xi,  xi ~ N(0, I),

which targets pi exactly only after a Metropolis accept/reject (MALA);
the unadjusted chain (ULA) is biased O(h^2). Gradient information lets
MALA scale as O(n^{1/3}) accept-optimal vs random-walk O(n^{-1}), so it
dominates RWM on smooth high-dimensional targets (Roberts & Rosenthal
1998 optimal-scaling theory).

Honesty: the bench verifies posterior moments on a correlated Gaussian
target (analytic truth) and confirms MALA's acceptance-rate advantage
over RWM at matched proposal scale. Fail-closed on non-finite density
or gradient.

References: Roberts & Tweedie (1996) "Exponential convergence of
Langevin distributions and their discrete approximations"; Roberts &
Rosenthal (1998) "Optimal scaling of discrete approximations to Langevin
diffusions"; Girolami & Calderhead (2011) RMHMC.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LogDensityFn = Callable[[FloatArray], float]
GradLogDensityFn = Callable[[FloatArray], FloatArray]


def _as_float(v: float) -> float:
    out = float(v)
    if not np.isfinite(out):
        raise ValueError("non-finite log density")
    return out


def mala(
    logp: LogDensityFn,
    grad_logp: GradLogDensityFn,
    x0: FloatArray,
    n_iter: int = 5000,
    step: float = 0.5,
    burn: int = 1000,
    seed: int = 0,
) -> tuple[FloatArray, float]:
    """MALA chain. Returns (post-burn draws, acceptance rate).

    Proposal: y = x + (step^2/2) g(x) + step * N(0, I); accepted with
    the standard Metropolis-Hastings log-ratio including the
    Langevin drift in both proposal densities.
    """
    if not (0.0 < step < 5.0):
        raise ValueError("step out of range")
    x = np.asarray(x0, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    d = x.size
    chain = np.empty((n_iter, d))
    lp = _as_float(logp(x))
    g = np.asarray(grad_logp(x), dtype=float).reshape(-1)
    n_acc = 0
    h2 = step * step / 2.0
    for i in range(n_iter):
        mean_xy = x + h2 * g
        y = mean_xy + step * rng.standard_normal(d)
        lp_y = _as_float(logp(y))
        g_y = np.asarray(grad_logp(y), dtype=float).reshape(-1)
        mean_yx = y + h2 * g_y
        # log q(y|x) and log q(x|y) up to the shared constant
        dq_fwd = np.sum((y - mean_xy) ** 2)
        dq_bwd = np.sum((x - mean_yx) ** 2)
        log_alpha = lp_y - lp + (dq_fwd - dq_bwd) / (2.0 * step * step)
        if np.log(rng.random()) < log_alpha:
            x, lp, g = y, lp_y, g_y
            n_acc += 1
        chain[i] = x
    return np.asarray(chain[burn:], dtype=np.float64), n_acc / n_iter


def ula(
    logp: LogDensityFn,
    grad_logp: GradLogDensityFn,
    x0: FloatArray,
    n_iter: int = 5000,
    step: float = 0.2,
    burn: int = 1000,
    seed: int = 0,
) -> FloatArray:
    """Unadjusted Langevin chain (no MH correction; biased O(step^2))."""
    if not (0.0 < step < 2.0):
        raise ValueError("step out of range")
    x = np.asarray(x0, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    chain = np.empty((n_iter, x.size))
    for i in range(n_iter):
        g = np.asarray(grad_logp(x), dtype=float).reshape(-1)
        x = x + (step * step / 2.0) * g + step * rng.standard_normal(x.size)
        chain[i] = x
    return np.asarray(chain[burn:], dtype=np.float64)


def rwmh(
    logp: LogDensityFn,
    x0: FloatArray,
    n_iter: int = 5000,
    step: float = 0.5,
    burn: int = 1000,
    seed: int = 0,
) -> tuple[FloatArray, float]:
    """Random-walk Metropolis baseline for the acceptance comparison."""
    x = np.asarray(x0, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    chain = np.empty((n_iter, x.size))
    lp = _as_float(logp(x))
    n_acc = 0
    for i in range(n_iter):
        y = x + step * rng.standard_normal(x.size)
        lp_y = _as_float(logp(y))
        if np.log(rng.random()) < lp_y - lp:
            x, lp = y, lp_y
            n_acc += 1
        chain[i] = x
    return np.asarray(chain[burn:], dtype=np.float64), n_acc / n_iter


def bench_mala(seed: int = 20261231 + 397) -> dict[str, float]:
    """SYNTHETIC check — MALA recovers correlated-Gaussian moments and
    beats RWMH acceptance at matched scale."""
    mu = np.array([1.5, -0.5])
    rho = 0.8
    cov = np.array([[1.0, rho], [rho, 1.0]])
    prec = np.linalg.inv(cov)

    def logp(x: FloatArray) -> float:
        d = x - mu
        return float(-0.5 * d @ prec @ d)

    def grad(x: FloatArray) -> FloatArray:
        return np.asarray(-(prec @ (x - mu)), dtype=np.float64)

    draws, acc_m = mala(logp, grad, np.zeros(2), n_iter=8000, step=0.6, burn=1500, seed=seed)
    m_err = float(np.linalg.norm(draws.mean(axis=0) - mu))
    cov_err = float(np.linalg.norm(np.cov(draws.T) - cov))
    if m_err > 0.15 or cov_err > 0.35:
        raise ValueError(f"MALA moments off: {m_err} {cov_err}")
    _, acc_r = rwmh(logp, np.zeros(2), n_iter=8000, step=0.6, burn=1500, seed=seed)
    if not (acc_m > acc_r):
        raise ValueError("MALA acceptance not beating RWM at same step")
    # ULA sanity: started at the posterior mean it stays close; the
    # mean residual is Monte-Carlo error of the autocorrelated chain
    # (discretization bias is O(step^2) and enters mostly in tails).
    du = ula(logp, grad, mu.copy(), n_iter=8000, step=0.2, burn=1500, seed=seed)
    u_err = float(np.linalg.norm(du.mean(axis=0) - mu))
    if u_err > 0.45:
        raise ValueError("ULA mean off")
    return {
        "synthetic_mala_mean_err": m_err,
        "synthetic_mala_cov_err": cov_err,
        "synthetic_mala_acc": float(acc_m),
        "synthetic_mala_rwm_acc": float(acc_r),
        "synthetic_mala_ula_mean_err": u_err,
        "score": 1.0,
    }
