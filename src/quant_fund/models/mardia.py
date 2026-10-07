"""Mardia's multivariate normality tests.

Mardia (1970): for a p-variate sample the multivariate skewness and
kurtosis measures

    b_1p = (1/n^2) sum_ij [(x_i - xbar)^T S^{-1} (x_j - xbar)]^3
    b_2p = (1/n)   sum_i  [(x_i - xbar)^T S^{-1} (x_i - xbar)]^2

reduce to the univariate measures at p=1 and are affine-invariant.
Under MVN, n b_1p/6 ~ chi^2[p(p+1)(p+2)/6] asymptotically and b_2p
is normal with mean p(p+2) and variance 8p(p+2)/n. Small-sample
corrections: Mardia's kurtosis z-statistic uses (b2p - p(p+2)) /
sqrt(8p(p+2)/n).

Honesty: the bench samples genuine MVN data (null respected at a
loose bound) and a Student-t(4) contaminant (rejected). The chi^2/
normal approximations are asymptotic — p bounds are generous and
documented. Fail-closed on singular covariance or tiny samples.

References: Mardia (1970) "Measures of multivariate skewness and
kurtosis with applications"; Mardia (1974) "Applications of some
measures of multivariate skewness and kurtosis"; Rencher &
Christensen (2012) ch. 4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

FloatArray = NDArray[np.float64]


def mardia_test(x: FloatArray) -> dict[str, float]:
    """Mardia skewness/kurtosis tests for multivariate normality.

    ``x`` is (n, p) with n >= 20 and p >= 2. Returns ``b1p``,
    ``b2p``, ``skew_p`` (chi^2), ``kurt_p`` (normal, two-sided).
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 20 or a.shape[1] < 2 or not np.isfinite(a).all():
        raise ValueError("bad sample")
    n, p = a.shape
    xc = a - a.mean(axis=0)
    s = xc.T @ xc / n
    try:
        sinv = np.linalg.inv(s)
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular covariance") from exc
    # mahalanobis inner products r_ij = xc_i^T S^-1 xc_j
    r = xc @ sinv @ xc.T
    b1p = float((r**3).sum() / (n * n))
    b2p = float(np.trace(r**2) / n) if False else float((r.diagonal() ** 2).sum() / n)
    # skewness: chi^2 with df = p(p+1)(p+2)/6
    df = p * (p + 1) * (p + 2) / 6.0
    skew_p = float(chi2.sf(n * b1p / 6.0, df))
    # kurtosis: z ~ N(p(p+2), 8p(p+2)/n)
    mu = p * (p + 2)
    z = (b2p - mu) / np.sqrt(8.0 * p * (p + 2) / n)
    kurt_p = float(2.0 * norm.sf(abs(z)))
    return {
        "b1p": b1p,
        "b2p": b2p,
        "skew_p": skew_p,
        "kurt_p": kurt_p,
        "z_kurt": float(z),
    }


def bench_mardia(seed: int = 20261231 + 426) -> dict[str, float]:
    """SYNTHETIC check — MVN null respected, heavy-tail rejected."""
    rng = np.random.default_rng(seed)
    n, p = 300, 3
    x = rng.standard_normal((n, p)) @ np.array([[1.0, 0.4, 0.1], [0.4, 1.2, 0.3], [0.1, 0.3, 0.8]])
    out_n = mardia_test(x)
    # Student-t(4) contaminant: excess kurtosis must reject
    xt = rng.standard_t(4.0, (n, p)) * np.array([1.0, 0.8, 0.6])
    out_t = mardia_test(xt)
    p_null = min(out_n["skew_p"], out_n["kurt_p"])
    p_alt = out_t["kurt_p"]
    if p_null < 0.01 or p_alt > 0.01:
        raise ValueError(f"mardia off: null={p_null:.4f} alt={p_alt:.4f}")
    return {
        "synthetic_mardia_p_null": p_null,
        "synthetic_mardia_p_alt": p_alt,
        "synthetic_mardia_b2p_null": out_n["b2p"],
        "synthetic_score": 1.0,
    }
