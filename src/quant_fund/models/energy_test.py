"""Szekely-Rizzo (2005) energy test for multivariate normality.

The energy distance between the standardized empirical
distribution and N(0, I_p) yields the test statistic

    E = n ( 2/n sum_j E||y_j - Z||
            - E||Z - Z'||
            - 1/n^2 sum_{i,j} ||y_i - y_j|| )

with y_j = S^{-1/2} (x_j - xbar) and, for Z ~ N(0, I_p),

    E||a - Z|| = sqrt(2) * G((p+1)/2)/G(p/2)
                 * 1F1(-1/2; p/2; -||a||^2 / 2),
    E||Z - Z'|| = 2 * G((p+1)/2)/G(p/2)

evaluated exactly via ``scipy.special.hyp1f1``. The
p-value comes from a parametric bootstrap under H0
(MVN draws with the fitted covariance).

Honesty: `bench_energy_test` checks the test accepts an
MVN sample and rejects a heavy-tailed t-mixture on
SYNTHETIC data — a size/power diagnostic, not a
real-data normality claim.

References
----------
* Szekely & Rizzo (2005) "A new test for multivariate
  normality", J Multivariate Analysis 93, 58-80.
* Rizzo & Szekely (2016) "Energy distance", WIREs Comput
  Stat 8, 27-38.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import special

FloatArray = NDArray[np.float64]


def _expected_z(a: FloatArray, p: int) -> FloatArray:
    """E||a - Z|| for Z ~ N(0, I_p) elementwise."""
    c = math.sqrt(2.0) * math.exp(special.gammaln((p + 1.0) / 2.0) - special.gammaln(p / 2.0))
    sq = np.sum(a * a, axis=1)
    return np.asarray(c * special.hyp1f1(-0.5, p / 2.0, -sq / 2.0), dtype=np.float64)


def _whiten(x: FloatArray) -> FloatArray:
    xc = x - x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + 1e-9 * np.eye(x.shape[1])
    # inverse symmetric sqrt via eigendecomposition
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    prec_sqrt = v @ np.diag(1.0 / np.sqrt(w)) @ v.T
    return np.asarray(xc @ prec_sqrt, dtype=np.float64)


def _energy_stat(y: FloatArray) -> float:
    n, p = y.shape
    term1 = 2.0 * float(_expected_z(y, p).mean())
    c = math.sqrt(2.0) * math.exp(special.gammaln((p + 1.0) / 2.0) - special.gammaln(p / 2.0))
    term2 = 2.0 * c  # E||Z - Z'|| = 2*sqrt(2)*G((p+1)/2)/G(p/2)
    diffs = y[:, None, :] - y[None, :, :]
    term3 = float(np.sqrt((diffs * diffs).sum(axis=2)).mean())
    return float(n * (term1 - term2 - term3))


def energy_mvn_test(x: FloatArray, n_boot: int = 300, seed: int = 0) -> dict[str, float]:
    """Energy statistic + parametric-bootstrap p-value."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] < 10 or xx.shape[1] < 1:
        raise ValueError("bad input matrix")
    n, p = xx.shape
    if n <= p + 5:
        raise ValueError("need n > p + 5")
    y = _whiten(xx)
    e_obs = _energy_stat(y)
    rng = np.random.default_rng(seed)
    count = 1
    for _ in range(n_boot):
        zb = rng.standard_normal((n, p))
        yb = _whiten(zb + xx.mean(axis=0))
        if _energy_stat(yb) >= e_obs:
            count += 1
    p_val = float(count / (n_boot + 1))
    return {"e_stat": e_obs, "p": p_val, "n_boot": float(n_boot)}


def bench_energy_test(seed: int = 20261231 + 473) -> dict[str, float]:
    """SYNTHETIC check — accepts MVN, rejects heavy-tail mix."""
    rng = np.random.default_rng(seed)
    n, p = 200, 3
    a = rng.normal(size=(p, p))
    cov = a @ a.T + np.eye(p)
    x_mvn = rng.multivariate_normal(np.zeros(p), cov, size=n)
    out_mvn = energy_mvn_test(x_mvn, n_boot=150, seed=seed)
    if out_mvn["p"] < 0.05:
        raise ValueError(f"rejected MVN: p={out_mvn['p']}")
    mix = x_mvn.copy()
    mask = rng.random(n) < 0.2
    mix[mask] = rng.standard_t(2.5, size=(int(mask.sum()), p)) * 2.5
    out_mix = energy_mvn_test(mix, n_boot=150, seed=seed)
    if out_mix["p"] > 0.05:
        raise ValueError(f"accepted mixture: p={out_mix['p']}")
    return {
        "synthetic_mvn_p": float(out_mvn["p"]),
        "synthetic_mix_p": float(out_mix["p"]),
        "synthetic_e_mvn": float(out_mvn["e_stat"]),
        "synthetic_e_mix": float(out_mix["e_stat"]),
        "synthetic_score": 1.0,
    }
