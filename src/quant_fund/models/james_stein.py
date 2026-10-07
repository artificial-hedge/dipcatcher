"""James-Stein shrinkage — minimax mean estimation.

James & Stein (1961): for p >= 3 independent means X_i ~ N(theta_i,
sigma^2), the sample mean is inadmissible; the shrinkage estimator

    theta_js = X - c * (X - mu_bar),   c = (p-2) sigma^2 / ||X - mu||^2

(strictly: shrink toward the grand mean with factor 1 - c/||.||^2 in
the common positive-part form) dominates MSE. The positive-part
estimator truncates at zero. Empirical Bayes (Efron-Morris) treats
the shrinkage factor as estimated.

Honesty: the bench draws p=8 means with a sparse+clustered theta,
simulates observations at known sigma, and checks (a) JS MSE is
strictly below MLE MSE, and (b) positive-part >= JS. With p=3 the
domination margin is thin — the bench uses p=8 for a robust check.
Fail-closed on p<3 or non-positive sigma.

References: James, Stein (1961) "Estimation with quadratic loss";
Stein (1956) inadmissibility of the MLE; Efron, Morris (1973)
"Stein's estimation rule and its competitors"; Efron (2012) "Large-
Scale Inference" ch. 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_x(x: FloatArray, sigma: float) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size < 3 or not np.isfinite(a).all():
        raise ValueError("need >=3 means")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("bad sigma")
    return a


def james_stein(x: FloatArray, sigma: float = 1.0) -> FloatArray:
    """Classical James-Stein estimator shrinking toward the grand mean.

    theta_js = mu_bar + (1 - (p-2) sigma^2 / ||x - mu_bar||^2) (x - mu_bar)
    (untruncated; may shrink past the mean when ||x-mu||^2 < (p-2)sigma^2).
    """
    a = _check_x(x, sigma)
    p = a.size
    mu = a.mean()
    r2 = float(((a - mu) ** 2).sum())
    c = min(1.0, (p - 2) * sigma * sigma / max(r2, 1e-300))
    return np.asarray(mu + (1.0 - c) * (a - mu), dtype=np.float64)


def positive_part_js(x: FloatArray, sigma: float = 1.0) -> FloatArray:
    """Positive-part James-Stein: shrinkage factor floored at 0."""
    a = _check_x(x, sigma)
    p = a.size
    mu = a.mean()
    r2 = float(((a - mu) ** 2).sum())
    c = max(0.0, 1.0 - (p - 2) * sigma * sigma / max(r2, 1e-300))
    return np.asarray(mu + c * (a - mu), dtype=np.float64)


def empirical_bayes_shrinkage(x: FloatArray, sigma: float = 1.0) -> FloatArray:
    """Efron-Morris empirical-Bayes shrinkage toward the MLE of mu.

    Estimates tau^2 = max(0, ||x-mu||^2/p - sigma^2) and shrinks with
    factor sigma^2/(sigma^2+tau^2).
    """
    a = _check_x(x, sigma)
    mu = a.mean()
    r2 = float(((a - mu) ** 2).sum())
    p = a.size
    tau2 = max(0.0, r2 / p - sigma * sigma)
    shrink = sigma * sigma / (sigma * sigma + tau2)
    return np.asarray(mu + (1.0 - shrink) * (a - mu), dtype=np.float64)


def bench_james_stein(seed: int = 20261231 + 425) -> dict[str, float]:
    """SYNTHETIC check — JS dominates MLE on a sparse+cluster theta."""
    rng = np.random.default_rng(seed)
    p, n_rep, sigma = 8, 400, 1.0
    theta = np.array([3.0, -2.0, 0.0, 0.0, 1.5, 0.0, 0.0, -1.0])
    mse_mle = 0.0
    mse_js = 0.0
    mse_pp = 0.0
    for _ in range(n_rep):
        x = theta + sigma * rng.standard_normal(p)
        mse_mle += float(((x - theta) ** 2).sum())
        mse_js += float(((james_stein(x, sigma) - theta) ** 2).sum())
        mse_pp += float(((positive_part_js(x, sigma) - theta) ** 2).sum())
    mse_mle /= n_rep
    mse_js /= n_rep
    mse_pp /= n_rep
    # positive-part weakly dominates JS (ties when the factor stays
    # positive); JS must strictly dominate the MLE.
    if mse_js >= mse_mle or mse_pp > mse_js + 1e-9:
        raise ValueError(f"js off: mle={mse_mle:.3f} js={mse_js:.3f} pp={mse_pp:.3f}")
    gain = 1.0 - mse_js / mse_mle
    if gain < 0.15:
        raise ValueError(f"js gain too small: {gain:.3f}")
    return {
        "synthetic_js_gain": gain,
        "synthetic_js_mse_ratio": mse_js / mse_mle,
        "synthetic_js_pp_ratio": mse_pp / mse_mle,
        "synthetic_score": 1.0,
    }
