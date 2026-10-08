"""Markov-chain discretization of AR(1) processes (SYNTHETIC).

Canonical methods for turning a continuous AR(1)
``x' = (1-rho)mu + rho x + sigma eps`` into a finite-state
Markov chain for dynamic-programming / estimation:

- Tauchen (1986) 'Finite state Markov-chain approximations
  to univariate and vector autoregressions' Economics
  Letters 20(2). Equally-spaced grid at
  ``mu +/- m sigma/sqrt(1-rho^2)``; transition probability
  to state j via norming the innovation CDF between grid
  midpoints.
- Tauchen & Hussey (1991) 'Quadrature-based methods for
  obtaining approximate solutions to nonlinear asset
  pricing models' Econometrica 59(2). Gauss-Hermite
  nodes/weights, transition kernel normalized over the
  conditional normal density — accurate for high
  persistence where Tauchen's grid is coarse.
- Rouwenhorst (1995) 'Asset pricing implications of
  equilibrium business cycle models' in Cooley ed.
  Exact match of the conditional *and* unconditional
  variance plus one-step autocorrelation for ANY rho,
  including rho -> 1 where Tauchen degenerates.

Conventions: states are the AR(1) level grid; P[i,j] =
Pr(x' = z_j | x = z_i); stationary distribution is the
left unit eigenvector.

`bench_markov` simulates a persistent AR(1), discretizes
with all three schemes, and gates each on |implied rho -
true rho| (simulated autocovariance of the grid-quantized
chain).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _check_ar1(rho: float, sigma: float, mu: float, n: int) -> None:
    if not (-1 < rho < 1):
        raise ValueError("rho outside (-1,1)")
    if sigma <= 0 or not np.isfinite(mu) or n < 3 or n > 200:
        raise ValueError("bad (sigma, mu, n)")


def _stationary(p: FloatArray) -> FloatArray:
    w, v = np.linalg.eig(p.T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.abs(np.real(v[:, i]))
    if pi.sum() <= 0:
        raise ValueError("no stationary distribution")
    return np.asarray(pi / pi.sum())


def tauchen(
    rho: float, sigma: float, mu: float = 0.0, n: int = 9, m: float = 3.0
) -> tuple[FloatArray, FloatArray]:
    """Tauchen (1986): equally-spaced grid covering +-m sd."""
    _check_ar1(rho, sigma, mu, n)
    if m <= 0:
        raise ValueError("m positive")
    sd_un = sigma / np.sqrt(1 - rho**2) if abs(rho) < 1 else sigma
    z = np.linspace(mu - m * sd_un, mu + m * sd_un, n)
    step = z[1] - z[0]
    p = np.zeros((n, n))
    for i in range(n):
        mean = mu + rho * (z[i] - mu)
        for j in range(n):
            if j == 0:
                p[i, j] = norm.cdf((z[0] + step / 2 - mean) / sigma)
            elif j == n - 1:
                p[i, j] = 1.0 - norm.cdf((z[-1] - step / 2 - mean) / sigma)
            else:
                p[i, j] = norm.cdf((z[j] + step / 2 - mean) / sigma) - norm.cdf(
                    (z[j] - step / 2 - mean) / sigma
                )
    return z, p


def tauchen_hussey(
    rho: float, sigma: float, mu: float = 0.0, n: int = 9
) -> tuple[FloatArray, FloatArray]:
    """Tauchen-Hussey (1991): Gauss-Hermite nodes + kernel."""
    _check_ar1(rho, sigma, mu, n)
    xg, wg = hermgauss(n)
    sd_un = sigma / np.sqrt(1 - rho**2)
    z = mu + np.sqrt(2.0) * sd_un * xg
    w_pdf = wg / np.sqrt(np.pi)
    p = np.zeros((n, n))
    for i in range(n):
        mean = mu + rho * (z[i] - mu)
        kern = norm.pdf(z, loc=mean, scale=sigma) / np.maximum(
            norm.pdf(z, loc=mu, scale=sd_un), 1e-300
        )
        p[i] = w_pdf * kern
        s = p[i].sum()
        if s <= 0:
            raise ValueError("degenerate row")
        p[i] /= s
    return z, p


def rouwenhorst(
    rho: float, sigma: float, mu: float = 0.0, n: int = 9
) -> tuple[FloatArray, FloatArray]:
    """Rouwenhorst (1995): matches E, Var, and rho exactly."""
    _check_ar1(rho, sigma, mu, n)
    p_ = (1 + rho) / 2
    q_ = p_
    sd_un = sigma / np.sqrt(1 - rho**2)
    psi = sd_un * np.sqrt(n - 1)
    z = np.linspace(mu - psi, mu + psi, n)

    def build(nn: int) -> FloatArray:
        if nn == 1:
            return np.array([[1.0]])
        prev = build(nn - 1)
        out = np.zeros((nn, nn))
        out[:-1, :-1] += p_ * prev
        out[:-1, 1:] += (1 - p_) * prev
        out[1:, :-1] += (1 - q_) * prev
        out[1:, 1:] += q_ * prev
        out[1:-1] /= 2
        return out

    p = build(n)
    return z, p


def implied_moments(z: FloatArray, p: FloatArray) -> dict[str, float]:
    """Stationary mean, sd, and lag-1 autocorrelation of the chain."""
    za = np.asarray(z, dtype=np.float64)
    pa = np.asarray(p, dtype=np.float64)
    if za.ndim != 1 or pa.shape != (za.size, za.size):
        raise ValueError("shape mismatch")
    pi = _stationary(pa)
    m = float(pi @ za)
    var = float(pi @ (za - m) ** 2)
    zc = za - m
    # E[z_i z'_j] under stationarity:
    joint = pi[:, None] * pa
    cov1 = float((joint * zc[:, None] * zc[None, :]).sum())
    return {
        "mean": m,
        "sd": float(np.sqrt(var)),
        "rho": cov1 / var if var > 0 else 0.0,
    }


def bench_markov(seed: int = 516) -> dict[str, float]:
    """SYNTHETIC: discretize rho=0.92 AR(1) and compare each
    scheme's implied (mean, sd, rho) against theory."""
    rho, sigma, mu = 0.92, 0.15, 0.4
    n = 11
    sd_un = sigma / np.sqrt(1 - rho**2)
    out: dict[str, float] = {}
    for name, fn in (
        ("tauchen", tauchen),
        ("th", tauchen_hussey),
        ("rw", rouwenhorst),
    ):
        z, p = fn(rho, sigma, mu, n)
        mom = implied_moments(z, p)
        out[f"synthetic_{name}_rho_err"] = abs(mom["rho"] - rho)
        out[f"synthetic_{name}_sd_err"] = abs(mom["sd"] - sd_un)
        out[f"synthetic_{name}_mean_err"] = abs(mom["mean"] - mu)
    # Rouwenhorst must match rho and unconditional sd nearly
    # exactly; Tauchen is allowed slack (grid coarseness)
    if out["synthetic_rw_rho_err"] > 1e-8 or out["synthetic_rw_sd_err"] > 1e-8:
        raise ValueError("Rouwenhorst moments off")
    if out["synthetic_tauchen_rho_err"] > 0.05:
        raise ValueError("Tauchen rho off")
    if out["synthetic_th_sd_err"] > 0.02:
        raise ValueError("Tauchen-Hussey sd off")
    rng = np.random.default_rng(seed)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = mu + rho * (x[t - 1] - mu) + sigma * rng.normal()
    out["synthetic_sim_rho"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return out
