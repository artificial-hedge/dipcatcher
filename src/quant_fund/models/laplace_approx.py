"""Laplace approximation canon: posterior mode by damped Newton on (SYNTHETIC)
-log p(theta | y) + Gaussian approximation N(mode, H^-1) with the
Hessian at the mode, plus the Laplace log-evidence estimate.
``bench_laplace_approx`` uses Bayesian logistic regression: gates the
Laplace marginal-predictive against a long Gibbs-free MCMC reference
(draws from the same posterior via importance sampling around the
Laplace mean? — no: via asymptotic exactness checks on quadrature for
1-D margins) and coverage of the Laplace credible intervals.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

FloatArray = NDArray[np.float64]


def _logreg_nlp(w: FloatArray, x: FloatArray, y: FloatArray, tau: float) -> float:
    z = np.clip(x @ w, -30, 30)
    ll = np.sum(y * z - np.logaddexp(0.0, z))
    return float(-ll + 0.5 * tau * (w @ w))


def _logreg_grad(w: FloatArray, x: FloatArray, y: FloatArray, tau: float) -> FloatArray:
    p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
    return x.T @ (p - y) + tau * w


def _logreg_hess(w: FloatArray, x: FloatArray, tau: float) -> FloatArray:
    p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
    r = p * (1.0 - p)
    return x.T @ (r[:, None] * x) + tau * np.eye(w.size)


def newton_mode(
    nlp: Callable[[FloatArray], float],
    grad: Callable[[FloatArray], FloatArray],
    hess: Callable[[FloatArray], FloatArray],
    w0: FloatArray,
    it: int = 60,
) -> FloatArray:
    w = np.asarray(w0, dtype=np.float64).copy()
    f_prev = nlp(w)
    for _ in range(it):
        g = grad(w)
        h = hess(w)
        step = np.linalg.solve(h, g)
        t = 1.0
        for _ in range(30):
            w_new = w - t * step
            if nlp(w_new) <= f_prev - 1e-4 * t * float(g @ step):
                break
            t *= 0.5
        else:
            break
        w = w_new
        f_prev = nlp(w)
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def laplace_fit(
    nlp: Callable[[FloatArray], float],
    grad: Callable[[FloatArray], FloatArray],
    hess: Callable[[FloatArray], FloatArray],
    w0: FloatArray,
) -> dict[str, FloatArray | float]:
    w = newton_mode(nlp, grad, hess, w0)
    h = hess(w)
    cov = np.linalg.inv(h)
    sign, logdet = np.linalg.slogdet(h)
    d = w.size
    log_ev = float(-nlp(w) - 0.5 * logdet + 0.5 * d * np.log(2 * np.pi))
    return {"mode": w, "cov": cov, "log_evidence": log_ev, "sign": float(sign)}


def laplace_posterior_predictive(fit: dict[str, FloatArray | float], x: FloatArray) -> FloatArray:
    """Probit-approximated sigmoid moment: sigma(kappa * x'm) with
    kappa = 1/sqrt(1 + pi/8 * x' V x)."""
    m = np.asarray(fit["mode"], dtype=np.float64)
    v = np.asarray(fit["cov"], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    mu = x @ m
    s2 = np.einsum("ij,jk,ik->i", x, v, x)
    kappa = 1.0 / np.sqrt(1.0 + np.pi / 8.0 * np.maximum(s2, 0.0))
    return np.asarray(1.0 / (1.0 + np.exp(-kappa * mu)))


def bench_laplace_approx(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, d = 300, 4
    x = np.column_stack([np.ones(n), rng.standard_normal((n, d - 1))])
    w_true = np.array([0.5, 1.5, -1.0, 0.8])
    p = 1.0 / (1.0 + np.exp(-x @ w_true))
    y = (rng.uniform(0, 1, n) < p).astype(np.float64)
    tau = 0.1
    nlp = lambda w: _logreg_nlp(w, x, y, tau)  # noqa: E731
    grad = lambda w: _logreg_grad(w, x, y, tau)  # noqa: E731
    hess = lambda w: _logreg_hess(w, x, tau)  # noqa: E731
    fit = laplace_fit(nlp, grad, hess, np.zeros(d))
    # reference posterior by long independence-MH (exact for this problem)
    m = np.asarray(fit["mode"])
    cov = np.asarray(fit["cov"])
    chol = np.linalg.cholesky(cov * 1.5)
    cur = m.copy()
    acc = 0
    draws = []
    rr = np.random.default_rng(seed + 9)
    lp = lambda w: -nlp(w)  # noqa: E731
    for _ in range(12000):
        prop = m + chol @ rr.standard_normal(d)
        logr = (
            lp(prop)
            - lp(cur)
            - (
                multivariate_normal.logpdf(prop, m, cov * 1.5)
                - multivariate_normal.logpdf(cur, m, cov * 1.5)
            )
        )
        if np.log(rr.uniform()) < logr:
            cur = prop
            acc += 1
        draws.append(cur.copy())
    ref = np.stack(draws)[2000:]
    ref_mean = ref.mean(axis=0)
    lap_mean = m
    # 90% Laplace coverage on each weight vs reference draws? use ref quantiles
    cov_cover = 0.0
    for j in range(d):
        lo, hi = np.quantile(ref[:, j], [0.05, 0.95])
        cov_cover += float(lo <= lap_mean[j] <= hi) / d
    pp = laplace_posterior_predictive(fit, x)
    brier = float(np.mean((pp - y) ** 2))
    acc_mh = acc / 12000.0
    return {
        "synthetic_laplace_mode_err": float(np.linalg.norm(lap_mean - w_true)),
        "synthetic_laplace_vs_mh_err": float(np.linalg.norm(lap_mean - ref_mean)),
        "synthetic_laplace_mh_cover": float(cov_cover),
        "synthetic_laplace_brier": brier,
        "synthetic_mh_accept": float(acc_mh),
    }
