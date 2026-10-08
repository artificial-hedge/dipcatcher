"""Henze-Zirkler (1990) multivariate normality test.

Assesses whether a multivariate sample is consistent with
a Gaussian copula+covariance using the kernel-weighted
empirical characteristic function. The statistic is

    HZ = (1/n^2) sum_ij exp(-b^2/2 D_ij)
         - 2 (1+b^2)^(-p/2) (1/n) sum_j exp(-b^2/(2(1+b^2)) D_j)
         + (1+2 b^2)^(-p/2)

with Mahalanobis distances D_j = (x_j-xbar)' S^-1 (x_j-xbar)
and D_ij likewise for pairs; the smoothing parameter

    b = (1/sqrt(2)) * ((2p+1)/4)^(1/(p+4)) * n^(1/(p+4))

is the Henze-Wagner choice. Under H0, HZ converges to a
lognormal with moments

    mu = 1 - a^(-p/2) (1 + p b^2/a + p(p+2) b^4/(2 a^2)),   a = 1+2b^2
    var = 2 (1+4b^2)^(-p/2)
          + 2 a^(-p) (1 + 2p b^4/a^2 + 3p(p+2) b^8/(4 a^4))
          - 4 w^(-p/2) (1 + 3p b^4/(2 w) + p(p+2) b^8/(2 w^2)),
    w = (1+b^2)(1+3b^2)

and the p-value is the lognormal survival probability.

Honesty: `bench_henze_zirkler` checks the test accepts
MVN data at a reasonable rate and rejects a heavy-tailed
contaminated mixture on SYNTHETIC data — a size/power
diagnostic, not a real-data normality claim.

References
----------
* Henze & Zirkler (1990) "A class of invariant consistent
  tests for multivariate normality", Commun Stat Theory
  Methods 19(10), 3595-3617.
* Henze & Wagner (1997) "A new approach to the BHEP tests
  for multivariate normality", J Mult Anal 62, 1-23.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def henze_zirkler(x: FloatArray) -> dict[str, float]:
    """HZ statistic + lognormal-approximation p-value."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] < 4 or xx.shape[1] < 1:
        raise ValueError("bad input matrix")
    n, p = xx.shape
    if n <= p + 2:
        raise ValueError("need n > p + 2")
    xc = xx - xx.mean(axis=0)
    cov = np.cov(xx, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    # shrink toward diagonal for stability
    cov = cov + 1e-10 * np.eye(p)
    prec = np.linalg.inv(cov)
    # Mahalanobis distances
    dj = np.einsum("ij,jk,ik->i", xc, prec, xc)
    dji = dj[:, None] + dj[None, :] - 2.0 * xc @ prec @ xc.T
    np.fill_diagonal(dji, 0.0)
    b = (
        (1.0 / math.sqrt(2.0))
        * ((2.0 * p + 1.0) / 4.0) ** (1.0 / (p + 4.0))
        * n ** (1.0 / (p + 4.0))
    )
    t1 = float(np.exp(-0.5 * b * b * np.clip(dji, 0.0, None)).mean())
    a1 = 1.0 + b * b
    t2 = 2.0 * a1 ** (-p / 2.0) * float(np.exp(-(b * b) / (2.0 * a1) * dj).mean())
    a = 1.0 + 2.0 * b * b
    hz = n * (t1 - t2 + a ** (-p / 2.0))
    # lognormal null moments
    mu = 1.0 - a ** (-p / 2.0) * (1.0 + p * b * b / a + p * (p + 2.0) * b**4 / (2.0 * a * a))
    w = (1.0 + b * b) * (1.0 + 3.0 * b * b)
    var = (
        2.0 * (1.0 + 4.0 * b * b) ** (-p / 2.0)
        + 2.0
        * a ** (-p)
        * (1.0 + 2.0 * p * b**4 / (a * a) + 3.0 * p * (p + 2.0) * b**8 / (4.0 * a**4))
        - 4.0
        * w ** (-p / 2.0)
        * (1.0 + 3.0 * p * b**4 / (2.0 * w) + p * (p + 2.0) * b**8 / (2.0 * w * w))
    )
    if var <= 0:
        raise ValueError("degenerate HZ null variance")
    sig2 = math.log(1.0 + var / (mu * mu))
    mu_ln = math.log(mu) - 0.5 * sig2
    if hz <= 0:
        pval = 1.0
    else:
        z = (math.log(hz) - mu_ln) / math.sqrt(sig2)
        pval = float(stats.norm.sf(z))
    return {"hz": float(hz), "mu_null": float(mu), "var_null": float(var), "p": pval}


def bench_henze_zirkler(seed: int = 20261231 + 470) -> dict[str, float]:
    """SYNTHETIC check — accepts MVN, rejects contaminated mix."""
    rng = np.random.default_rng(seed)
    n, p = 400, 3
    a_mat = rng.normal(size=(p, p))
    cov = a_mat @ a_mat.T + np.eye(p)
    x_mvn = rng.multivariate_normal(np.zeros(p), cov, size=n)
    out_mvn = henze_zirkler(x_mvn)
    if out_mvn["p"] < 0.05:
        raise ValueError(f"rejected MVN: p={out_mvn['p']}")
    # 15% heavy-tail contamination
    mix = x_mvn.copy()
    mask = rng.random(n) < 0.15
    mix[mask] = rng.standard_t(2.0, size=(int(mask.sum()), p)) * 3.0
    out_mix = henze_zirkler(mix)
    if out_mix["p"] > 0.05:
        raise ValueError(f"accepted mixture: p={out_mix['p']}")
    return {
        "synthetic_mvn_p": float(out_mvn["p"]),
        "synthetic_mix_p": float(out_mix["p"]),
        "synthetic_hz_mvn": float(out_mvn["hz"]),
        "synthetic_hz_mix": float(out_mix["hz"]),
        "synthetic_score": 1.0,
    }
