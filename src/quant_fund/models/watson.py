"""Watson (1961) U^2 test for circular uniformity.

The U^2 statistic is the origin-invariant analogue of
Cramer-von Mises for data on the circle: for ordered
transformed observations z_(1) <= ... <= z_(n) on [0,1)
(under H0, z_j = F0(theta_j)),

    U^2 = sum_j (z_j - zbar - (j-0.5)/n + 0.5)^2
          - n (zbar - 0.5)^2 + 1/(12n)

with zbar the sample mean. Stephens (1970) provides the
modified statistic U*^2 = (U^2 - 0.1/n + 0.1/n^2)(1 +
0.8/n) with the standard critical table.

Also returns Rayleigh's mean resultant length R/n and its
large-sample p-value for a unimodal alternative.

Honesty: `bench_watson` checks U^2 accepts uniform angles
and rejects a von Mises concentration on SYNTHETIC data —
a size/power diagnostic, not a real-data claim.

References
----------
* Watson, G.S. (1961) "Goodness-of-fit tests on a
  circle", Biometrika 48, 109-114.
* Stephens, M.A. (1970) "Use of the Kolmogorov-Smirnov,
  Cramer-von Mises and related statistics without
  extensive tables", JRSS-B 32.
* Mardia & Jupp (2000) "Directional Statistics", ch. 6.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Stephens (1970) critical table for modified U^2
_U2_CRIT = np.array([[0.10, 0.05, 0.025, 0.01], [0.152, 0.187, 0.221, 0.267]])


def _u2_pvalue(u2_mod: float) -> float:
    """Interpolate the Stephens table (log-linear in alpha)."""
    alphas, crits = _U2_CRIT[0], _U2_CRIT[1]
    if u2_mod <= crits[0]:
        return float(alphas[0] + (crits[0] - u2_mod) / crits[0] * (1.0 - alphas[0]))
    if u2_mod >= crits[-1]:
        return float(alphas[-1] * crits[-1] / u2_mod)
    i = int(np.searchsorted(crits, u2_mod))
    # log-linear interpolation in alpha
    la0, la1 = math.log(alphas[i - 1]), math.log(alphas[i])
    c0, c1 = crits[i - 1], crits[i]
    return float(math.exp(la0 + (u2_mod - c0) / (c1 - c0) * (la1 - la0)))


def watson_u2(theta: FloatArray) -> dict[str, float]:
    """Watson U^2 uniformity test on angles (radians)."""
    th = np.asarray(theta, dtype=np.float64).ravel()
    n = th.size
    if n < 8 or not np.all(np.isfinite(th)):
        raise ValueError("bad input")
    z = np.sort((th % (2.0 * math.pi)) / (2.0 * math.pi))
    j = np.arange(1, n + 1)
    w_ = z - (j - 0.5) / n
    u2 = float(((w_ - w_.mean()) ** 2).sum() + 1.0 / (12.0 * n))
    u2_mod = (u2 - 0.1 / n + 0.1 / (n * n)) * (1.0 + 0.8 / n)
    # Rayleigh mean resultant length
    r = float(math.hypot(float(np.cos(th).mean()), float(np.sin(th).mean())))
    z_r = n * r * r
    p_ray = float(
        math.exp(-z_r)
        * (
            1.0
            + (2.0 * z_r - z_r * z_r) / (4.0 * n)
            - (24.0 * z_r - 132.0 * z_r * z_r + 76.0 * z_r**3 - 9.0 * z_r**4) / (288.0 * n * n)
        )
    )
    return {
        "u2": u2,
        "u2_modified": float(u2_mod),
        "p_u2": _u2_pvalue(u2_mod),
        "rayleigh_r": r,
        "rayleigh_p": float(min(max(p_ray, 0.0), 1.0)),
    }


def bench_watson(seed: int = 20261231 + 472) -> dict[str, float]:
    """SYNTHETIC check — accepts uniform, rejects von Mises."""
    rng = np.random.default_rng(seed)
    n = 300
    u = rng.uniform(0.0, 2.0 * math.pi, size=n)
    out_u = watson_u2(u)
    if out_u["p_u2"] < 0.05:
        raise ValueError(f"rejected uniform: p={out_u['p_u2']}")
    vm = rng.vonmises(0.0, 3.0, size=n)
    out_v = watson_u2(vm)
    if out_v["p_u2"] > 0.05:
        raise ValueError(f"accepted von Mises: p={out_v['p_u2']}")
    if out_v["rayleigh_p"] > 0.05:
        raise ValueError(f"rayleigh missed: {out_v['rayleigh_p']}")
    return {
        "synthetic_uniform_p": float(out_u["p_u2"]),
        "synthetic_vm_p": float(out_v["p_u2"]),
        "synthetic_vm_rayleigh_p": float(out_v["rayleigh_p"]),
        "synthetic_score": 1.0,
    }
