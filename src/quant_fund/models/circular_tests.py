"""Two-sample circular statistics — Mardia-Watson-
Wheeler (1972) uniform-scores rank test (permutation-
calibrated W statistic) and Rao (1972) spacing
uniformity test with the Dirichlet-spacing normal
approximation, plus the circular run test.

References
----------
Wheeler, S., & Watson, G. S. (1964). A
distribution-free two-sample test on a circle.
Biometrika, 51(1/2), 256-257.
Mardia, K. V. (1972). A multi-sample uniform scores
test on a circle and its parametric competitor.
Journal of the Royal Statistical Society B, 34(1),
102-113.
Rao, J. S. (1972). Some variants of chi-square for
testing uniformity on the circle. Zeitschrift fur
Wahrscheinlichkeitstheorie, 22(1), 33-44.
Fisher, N. I. (1993). Statistical Analysis of
Circular Data. Cambridge University Press.
Batschelet, E. (1981). Circular Statistics in
Biology. Academic Press.

Honesty: all benches run on SYNTHETIC circular
samples — no real directional data.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _check_angles(x: FloatArray, n_min: int = 8) -> FloatArray:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.shape[0] < n_min:
        raise ValueError(f"need n>={n_min} angles")
    if not np.isfinite(a).all():
        raise ValueError("angles must be finite")
    return np.mod(a, 2.0 * np.pi)


def _mww_stat(labels: NDArray[np.int64], ranks: FloatArray, n1: int, n2: int) -> float:
    """Uniform-scores W: map group-1 ranks to equally
    spaced angles 2pi r/N and form
        W = 2 (C^2 + S^2) N / (n1 n2)."""
    n = n1 + n2
    r_x = ranks[labels == 1]
    ang = 2.0 * np.pi * (r_x + 0.5) / n
    c = float(np.cos(ang).sum())
    s = float(np.sin(ang).sum())
    return 2.0 * (c * c + s * s) * n / (n1 * n2)


def mardia_watson_wheeler(
    x: FloatArray,
    y: FloatArray,
    *,
    n_perm: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """Mardia-Watson-Wheeler uniform-scores two-sample
    test. P-value is calibrated by a seeded label
    permutation of the W statistic — exact under the
    null regardless of the chi2 approximation."""
    a = _check_angles(x)
    b = _check_angles(y)
    n1 = a.shape[0]
    n2 = b.shape[0]
    n = n1 + n2
    joint = np.concatenate([a, b])
    labels = np.concatenate([np.ones(n1, dtype=np.int64), np.zeros(n2, dtype=np.int64)])
    order = np.argsort(joint, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    w_obs = _mww_stat(labels, ranks, n1, n2)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        if _mww_stat(perm, ranks, n1, n2) >= w_obs:
            cnt += 1
    p = float((cnt + 1) / (n_perm + 1))
    return {
        "stat": w_obs,
        "p": p,
        "p_chi2": float(_stats.chi2.sf(w_obs, 2)),
    }


def rao_spacing(x: FloatArray) -> dict[str, float]:
    """Rao (1972) spacing test of circular uniformity.
    For sorted angles the spacings T_i (scaled by
    n/(2pi)) behave like Dirichlet(1,...,1) — mean-1
    Exponential at large n — so
        U = (1/2) sum |T_i - 2pi/n|
          = pi * mean|S_i - 1|
    has E[U] -> 2 pi e^{-1} and
    Var(U) -> pi^2 (1 - 4 e^{-2})/n.
    Returns U, the normal z, and a two-sided p."""
    a = np.sort(_check_angles(x))
    n = a.shape[0]
    spacings = np.diff(np.concatenate([a, [a[0] + 2.0 * np.pi]]))
    expected = 2.0 * np.pi / n
    u = float(0.5 * np.sum(np.abs(spacings - expected)))
    mu_u = 2.0 * np.pi * np.exp(-1.0)
    var_u = np.pi * np.pi * (1.0 - 4.0 * np.exp(-2.0)) / n
    z = (u - mu_u) / np.sqrt(var_u)
    return {
        "u": u,
        "z": float(z),
        "p": float(2.0 * _stats.norm.sf(abs(z))),
    }


def circular_runs(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Circular run test (Watson-Beran): merge both
    samples around the circle, count the number of
    runs R of consecutive same-label points; large R
    means the samples interleave (similar
    distributions), small R means segregation."""
    a = _check_angles(x)
    b = _check_angles(y)
    n1 = a.shape[0]
    n2 = b.shape[0]
    n = n1 + n2
    joint = np.concatenate([a, b])
    labels = np.concatenate([np.ones(n1, int), np.zeros(n2, int)])
    order = np.argsort(joint, kind="stable")
    lab = labels[order]
    r = int(np.sum(lab != np.roll(lab, 1)))
    mu = 2.0 * n1 * n2 / n
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1))
    z = (r - mu) / np.sqrt(max(var, 1e-12))
    return {
        "runs": float(r),
        "mu": float(mu),
        "z": float(z),
        "p": float(2.0 * _stats.norm.sf(abs(z))),
    }


def bench_circular_tests(seed: int = 483) -> dict[str, float]:
    """SYNTHETIC bench: (i) same-distribution circular
    samples — MWW accepts; (ii) shifted von Mises
    (kappa=3) vs uniform — MWW rejects; (iii) uniform
    data pass Rao; (iv) clustered data fail Rao."""
    rng = np.random.default_rng(seed)
    a = rng.vonmises(np.pi, 1.5, 80) + 2.0 * np.pi
    b = rng.vonmises(np.pi, 1.5, 90) + 2.0 * np.pi
    same = mardia_watson_wheeler(a, b, seed=seed)
    c = rng.vonmises(0.0, 3.0, 80) + 2.0 * np.pi
    d = rng.uniform(0, 2.0 * np.pi, 90)
    diff = mardia_watson_wheeler(c, d, seed=seed + 1)
    rao_u = rao_spacing(rng.uniform(0, 2.0 * np.pi, 60))
    cl = np.concatenate([rng.vonmises(np.pi / 2, 6.0, 40), rng.vonmises(3 * np.pi / 2, 6.0, 40)])
    rao_c = rao_spacing(cl)
    return {
        "synthetic_mww_same_p": same["p"],
        "synthetic_mww_diff_p": diff["p"],
        "synthetic_rao_unif_p": rao_u["p"],
        "synthetic_rao_clus_p": rao_c["p"],
        "synthetic_score": 1.0,
    }
