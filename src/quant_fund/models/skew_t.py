"""Hansen (1994) skewed Student-t distribution (SYNTHETIC).

Hansen's standardised skew-t has zero mean, unit variance, degrees of freedom
``nu > 2`` and skewness ``lambda in (-1, 1)``.  With

    c = Gamma((nu+1)/2) / (sqrt(pi (nu-2)) Gamma(nu/2)),
    a = 4 lambda c (nu-2)/(nu-1),   b = sqrt(1 + 3 lambda^2 - a^2),

the density of the standardised variable ``z`` is

    f(z) = b c (1 + (1/(nu-2)) ((b z + a)/(1 -/+ lambda))^2)^{-(nu+1)/2},

using ``1 - lambda`` for ``z < -a/b`` and ``1 + lambda`` otherwise.  The
quantile inverts that splice in closed form (one Student-t ``stdtrit`` per
probability), not a scalar root find.  A location-scale version
``x = mu + sigma z`` is fitted by maximum likelihood.

Reference: B. E. Hansen (1994), "Autoregressive conditional density
estimation", International Economic Review.  Fail-closed on invalid parameters
or non-finite input.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln, stdtr, stdtrit

Array = NDArray[np.float64]


def _abc(nu: float, lam: float) -> tuple[float, float, float]:
    if nu <= 2.0 or not -1.0 < lam < 1.0:
        raise ValueError("require nu > 2 and lambda in (-1, 1)")
    log_c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
    c = float(np.exp(log_c))
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b = float(np.sqrt(1.0 + 3.0 * lam**2 - a**2))
    return a, b, c


def skew_t_pdf(x: Array, nu: float, lam: float, mu: float = 0.0, sigma: float = 1.0) -> Array:
    """Hansen skew-t density (location-scale)."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    a, b, c = _abc(nu, lam)
    z = (np.asarray(x, dtype=float) - mu) / sigma
    denom = np.where(z < -a / b, 1.0 - lam, 1.0 + lam)
    kernel = 1.0 + (1.0 / (nu - 2.0)) * ((b * z + a) / denom) ** 2
    return b * c * kernel ** (-(nu + 1.0) / 2.0) / sigma


def skew_t_cdf(x: Array, nu: float, lam: float, mu: float = 0.0, sigma: float = 1.0) -> Array:
    """Hansen skew-t CDF via the standard Student-t CDF of each spliced branch.

    ``stdtr`` is the same primitive ``scipy.stats.t.cdf`` uses. The branch
    point ``z = -a/b`` has CDF ``(1-λ)/2``.
    """
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("sigma must be positive")
    a, b, _c = _abc(nu, lam)
    z = (np.asarray(x, dtype=float) - mu) / sigma
    t_scale = math.sqrt(nu / (nu - 2.0))
    lo = (1.0 - lam) * stdtr(nu, t_scale * (b * z + a) / (1.0 - lam))
    hi = (1.0 - lam) * 0.5 + (1.0 + lam) * (stdtr(nu, t_scale * (b * z + a) / (1.0 + lam)) - 0.5)
    return np.asarray(np.where(z < -a / b, lo, hi), dtype=float)


def _hansen_unit_quantile(p: Array, nu: float, lam: float) -> Array:
    """Quantile of Hansen's zero-mean, unit-variance skew-t.

    Left of the kink (``p <= (1-λ)/2``) the standard-t probability is
    ``p/(1-λ)``; on the right it is ``(p+λ)/(1+λ)``. Then

    ``z = (t_ν^{-1}(F) * (1∓λ) / sqrt(ν/(ν-2)) - a) / b``.
    """
    a, b, _c = _abc(nu, lam)
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError("skew-t scale factor b is not positive")
    t_scale = math.sqrt(nu / (nu - 2.0))
    left = p <= (0.5 * (1.0 - lam))
    one_m = 1.0 - lam
    one_p = 1.0 + lam
    prob = np.empty(p.shape, dtype=float)
    prob[left] = p[left] / one_m
    right = ~left
    prob[right] = (p[right] + lam) / one_p
    u = np.asarray(stdtrit(nu, prob), dtype=float)
    branch = np.where(left, one_m, one_p)
    return (u * branch / t_scale - a) / b


def skew_t_ppf(
    p: float | Array,
    nu: float,
    lam: float,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> float | Array:
    """Hansen skew-t quantile. Closed form of :func:`skew_t_cdf`.

    Scalar ``p`` returns ``float``. An array returns an array of the same
    shape. This replaces a Brent root find (one CDF inversion per probability)
    with one ``stdtrit`` evaluation. Research-diagnostic only — not a live
    capital claim.
    """
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("sigma must be positive")
    p_arr = np.asarray(p, dtype=float)
    scalar = bool(p_arr.ndim == 0)
    flat = np.reshape(p_arr, -1)
    if flat.size and not bool(np.all(np.isfinite(flat) & (flat > 0.0) & (flat < 1.0))):
        raise ValueError("p must be in (0, 1)")
    if flat.size == 0:
        return np.asarray(flat, dtype=float).reshape(p_arr.shape)
    out = mu + sigma * _hansen_unit_quantile(flat, nu, lam)
    if scalar:
        return float(out[0])
    return np.asarray(out, dtype=float).reshape(p_arr.shape)


def skew_t_fit(x: Array) -> dict[str, float]:
    """Maximum-likelihood fit returning ``nu``, ``lambda``, ``mu``, ``sigma``."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 30 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 30 observations")
    mu0, sigma0 = float(arr.mean()), float(arr.std(ddof=1))
    if sigma0 <= 0.0:
        raise ValueError("data has zero variance")

    def nll(theta: Array) -> float:
        nu, lam, mu, log_sigma = theta
        if nu <= 2.01 or not -0.98 < lam < 0.98:
            return 1e12
        pdf = skew_t_pdf(arr, float(nu), float(lam), float(mu), float(np.exp(log_sigma)))
        if not np.all(pdf > 0) or not np.all(np.isfinite(pdf)):
            return 1e12
        return -float(np.log(pdf).sum())

    x0 = np.array([8.0, 0.0, mu0, np.log(sigma0)])
    bounds = [
        (2.05, 200.0),
        (-0.97, 0.97),
        (mu0 - 5 * sigma0, mu0 + 5 * sigma0),
        (np.log(sigma0) - 3, np.log(sigma0) + 3),
    ]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    if not np.isfinite(res.fun) or res.fun >= 1e12:
        raise ValueError("skew-t MLE did not converge to a finite likelihood")
    nu, lam, mu, log_sigma = res.x
    return {
        "nu": float(nu),
        "lam": float(lam),
        "mu": float(mu),
        "sigma": float(np.exp(log_sigma)),
        "loglik": float(-res.fun),
    }
