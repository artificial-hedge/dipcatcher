"""Natural gradient descent on a Gaussian likelihood problem (SYNTHETIC).

Fit N(mu, sigma^2) to samples by descending expected negative log-likelihood
in the Fisher metric: update (mu, nu=log sigma) with F^{-1} grad where
F = diag(1/sigma^2, 2). Compared against vanilla Euclidean gradient descent
on the same objective — the honest gap is iteration count to tolerance.
"""

import numpy as np

from quant_fund.models._ig_synth import gauss_fit


def _objective(x: np.ndarray, mu: float, nu: float) -> float:
    sig2 = np.exp(2 * nu)
    return float(0.5 * np.mean((x - mu) ** 2) / sig2 + nu)


def _grads(x: np.ndarray, mu: float, nu: float) -> tuple[float, float]:
    sig2 = np.exp(2 * nu)
    g_mu = float(np.mean(mu - x) / sig2)
    g_nu = float(1.0 - np.mean((x - mu) ** 2) / sig2)
    return g_mu, g_nu


def _run(x: np.ndarray, natural: bool, lr: float, tol: float = 1e-6, maxit: int = 20000) -> int:
    mu, nu = 0.0, np.log(3.0)
    prev = _objective(x, mu, nu)
    for it in range(1, maxit + 1):
        g_mu, g_nu = _grads(x, mu, nu)
        if natural:
            sig2 = np.exp(2 * nu)
            step_mu = sig2 * g_mu
            step_nu = 0.5 * g_nu
        else:
            step_mu, step_nu = g_mu, g_nu
        mu -= lr * step_mu
        nu -= lr * step_nu
        cur = _objective(x, mu, nu)
        if abs(prev - cur) < tol:
            return it
        prev = cur
    return maxit


def bench_natural_gradient(seed: int = 4303) -> dict[str, float]:
    x = gauss_fit(seed)
    it_ng = _run(x, natural=True, lr=0.05)
    it_gd = _run(x, natural=False, lr=0.05)
    return {
        "synthetic_ng_iters": float(it_ng),
        "synthetic_gd_iters": float(it_gd),
        "synthetic_ng_speedup": float(it_gd / max(it_ng, 1)),
    }
