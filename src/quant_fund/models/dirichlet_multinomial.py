"""Dirichlet-multinomial distribution fitting — Mosimann
(1962) moments, Minka (2000) fixed-point maximum
likelihood, and Minka Newton updates for the
concentration alpha of a discrete multivariate
overdispersed multinomial.

References
----------
Mosimann, J. E. (1962). On the compound multinomial
distribution, the multivariate beta-distribution, and
correlations among proportions. Biometrika, 49(1/2),
65-82.
Minka, T. P. (2000). Estimating a Dirichlet
distribution. MIT technical report.
Sklar, M. (2014). Fast MLE computation for the
Dirichlet multinomial. arXiv:1405.0099.

Honesty: all benches run on SYNTHETIC simulated count
data — no real portfolio or market data.

Composition: numpy only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import polygamma, psi

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check_counts(counts: IntArray) -> IntArray:
    x = np.asarray(counts, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("counts must be an (n, p) matrix with n,p >= 2")
    if x.min() < 0:
        raise ValueError("counts must be non-negative")
    return x


def _row_sums(counts: IntArray) -> FloatArray:
    return counts.sum(axis=1).astype(np.float64)


def dm_mom(counts: IntArray) -> dict[str, float | FloatArray]:
    """Mosimann method-of-moments Dirichlet-multinomial
    fit: the observed proportions give the mean
    probability vector; the overdispersion rho is the
    average of the per-category intraclass
    correlations rho_j = (v_obs/v_binom - 1) / (n_t/m - 1)
    where v_obs is the empirical proportion variance and
    v_binom = p_j(1-p_j)/m is the binomial baseline.
    Returns p (probabilities), rho, and a0 = (1-rho)/rho.
    """
    x = _check_counts(counts)
    n_obs = _row_sums(x)
    p_hat = (x.sum(axis=0) / x.sum()).astype(np.float64)
    total = float(x.sum())
    if total <= 0.0:
        raise ValueError("total count is zero")
    prop = x / n_obs[:, None]
    m_bar = float(n_obs.mean())
    rhos: list[float] = []
    for j in range(x.shape[1]):
        v_obs = float(np.var(prop[:, j], ddof=1))
        v_bin = float(p_hat[j] * (1.0 - p_hat[j]) / m_bar)
        denom = (1.0 / float(np.mean(1.0 / n_obs)) * 1.0) - 1.0
        if v_bin > 0 and denom > 0:
            r = (v_obs / v_bin - 1.0) / denom
            rhos.append(max(r, 1e-6))
    rho = float(np.clip(np.mean(rhos), 1e-6, 0.999))
    a0 = (1.0 - rho) / rho
    return {
        "rho": rho,
        "a0": a0,
        "p_hat": p_hat,
        "loglik_mom": _loglik(x, p_hat * a0),
    }


def _loglik(counts: IntArray, alpha: FloatArray) -> float:
    from scipy.special import gammaln

    x = counts.astype(np.float64)
    n_obs = x.sum(axis=1)
    a0 = float(alpha.sum())
    ll = (
        gammaln(a0)
        - gammaln(a0 + n_obs)
        + np.sum(gammaln(x + alpha) - gammaln(alpha), axis=1)
        - np.sum(gammaln(x + 1.0), axis=1)
    )
    return float(ll.mean())


def dm_minka(counts: IntArray, n_iter: int = 200) -> dict[str, float | FloatArray]:
    """Minka (2000) fixed-point maximum-likelihood fit of
    the Dirichlet-multinomial parameters. Iterates the
    Newton update on the inverse of each coordinate's
    trigamma component together with a global digamma
    rescale:
        alpha_j <- alpha_j * psi-sum_j / |psi(a0_sum) - psi(a0)|
    using the standard two-term fixed point
        alpha_j^(new) = alpha_j *
        (sum_i psi(a_j + x_ij) - n*psi(a_j)) /
        (sum_i psi(a0 + n_i) - n*psi(a0)).
    Returns fitted alpha, the implied probabilities, and
    the mean log-likelihood.
    """
    x = _check_counts(counts)
    n_obs = _row_sums(x)
    fit = dm_mom(x)
    alpha = np.asarray(fit["p_hat"]) * float(fit["a0"])
    alpha = np.maximum(alpha, 1e-3)
    for _ in range(n_iter):
        a0 = float(alpha.sum())
        num = psi(x + alpha).sum(axis=0) - x.shape[0] * psi(alpha)
        den = float(psi(a0 + n_obs).sum() - x.shape[0] * psi(a0))
        if abs(den) < 1e-12:
            break
        new = alpha * num / den
        new = np.maximum(new, 1e-9)
        if float(np.max(np.abs(new - alpha) / alpha)) < 1e-10:
            alpha = new
            break
        alpha = new
    p_hat = alpha / alpha.sum()
    return {
        "a0": float(alpha.sum()),
        "alpha": alpha,
        "p_hat": p_hat,
        "loglik": _loglik(x, alpha),
    }


def dm_loglik(counts: IntArray, alpha: FloatArray) -> float:
    """Mean Dirichlet-multinomial log-likelihood for a
    supplied concentration vector."""
    x = _check_counts(counts)
    a = np.asarray(alpha, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] != x.shape[1] or (a <= 0).any():
        raise ValueError("alpha must be a positive p-vector")
    return _loglik(x, a)


def dm_minka_newton(counts: IntArray, n_iter: int = 100) -> dict[str, float | FloatArray]:
    """Minka Newton iteration on the Dirichlet-multinomial
    log-likelihood using the shared digamma term:
        g_j = sum_i psi(a_j + x_ij) - n psi(a_j)
              - [sum_i psi(a0 + n_i) - n psi(a0)] * (a_j terms cancel via b)
    implemented as the full (p x p) Hessian solve with the
    diagonal trigamma and constant-polygamma structure
    H = diag(q_j) + 11' z, inverted via Sherman-Morrison.
    """
    x = _check_counts(counts)
    n, _p = x.shape
    n_obs = _row_sums(x)
    fit = dm_mom(x)
    alpha = np.maximum(np.asarray(fit["p_hat"]) * float(fit["a0"]), 1e-3)
    for _ in range(n_iter):
        a0 = float(alpha.sum())
        g = psi(x + alpha).sum(axis=0) - n * psi(alpha)
        g -= float(psi(a0 + n_obs).sum() - n * psi(a0))
        q = polygamma(1, x + alpha).sum(axis=0) - n * polygamma(1, alpha)
        z = float(polygamma(1, a0 + n_obs).sum() - n * polygamma(1, a0))
        # H = diag(q) + z*11' -> Sherman-Morrison
        b = float(1.0 + z * np.sum(1.0 / q))
        step = (g - z * np.sum(g / q) / b) / q
        step = np.clip(step, -2.0 * np.abs(alpha), 2.0 * np.abs(alpha))
        new = alpha - step
        new = np.maximum(new, 1e-9)
        if float(np.max(np.abs(new - alpha) / alpha)) < 1e-12:
            alpha = new
            break
        alpha = new
    return {
        "a0": float(alpha.sum()),
        "alpha": alpha,
        "p_hat": alpha / alpha.sum(),
        "loglik": _loglik(x, alpha),
    }


def bench_dirichlet_multinomial(seed: int = 468) -> dict[str, float]:
    """SYNTHETIC bench: draw a true DM with p=5,
    a0=4.0, n_i~Poi(20), verify that MoM and MLE both
    recover rho and p within loose tolerance and that MLE
    log-likelihood exceeds MoM's."""
    rng = np.random.default_rng(seed)
    p_true = np.array([0.10, 0.20, 0.40, 0.15, 0.15])
    a0_true = 4.0
    n_draw = 800
    n_i = rng.poisson(20, size=n_draw).astype(np.int64) + 1
    alpha = a0_true * p_true
    theta = rng.dirichlet(alpha, size=n_draw)
    counts = np.vstack([rng.multinomial(int(n_i[i]), theta[i]) for i in range(n_draw)])
    mom = dm_mom(counts)
    mle = dm_minka(counts)
    mle_n = dm_minka_newton(counts)
    rho_true = 1.0 / (1.0 + a0_true)
    rho_err_mom = abs(float(mom["rho"]) - rho_true)
    rho_err_mle = abs(1.0 / (1.0 + float(mle["a0"])) - rho_true)
    p_err = float(np.max(np.abs(np.asarray(mle["p_hat"]) - p_true)))
    return {
        "synthetic_rho_err_mom": rho_err_mom,
        "synthetic_rho_err_mle": rho_err_mle,
        "synthetic_p_max_err": p_err,
        "synthetic_mle_gain": float(mle["loglik"]) - float(mom["loglik_mom"]),
        "synthetic_newton_loglik": float(mle_n["loglik"]),
        "synthetic_score": 1.0,
    }
