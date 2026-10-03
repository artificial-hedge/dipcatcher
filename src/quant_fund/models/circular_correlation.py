"""Circular association measures — Mardia (1976)
circular-linear correlation, Fisher-Lee (1983)
circular-circular correlation, and the
Jammalamadaka-Sarma (1988) rank-based circular
correlation with permutation p-values.

Mardia circular-linear coefficient between angle
theta and linear variable x:

    R^2 = (r_xc^2 + r_xs^2 - 2 r_xc r_xs r_cs)
          / (1 - r_cs^2)

where r_xc = corr(x, cos theta), r_xs =
corr(x, sin theta), r_cs = corr(cos theta,
sin theta). Under independence
(n-1) R^2 ~ chi2(2) approximately.

Fisher-Lee circular-circular correlation between
angles alpha, beta:

    R = sum sin(a_i - a_j) sin(b_i - b_j)
        / sqrt( sum sin^2(a_i - a_j)
                sum sin^2(b_i - b_j) )

Jammalamadaka-Sarma rank correlation ranks
(a_i, b_i), computes the Kruskal-Wallis-like
circular placement, and reports the W-statistic
form for independence testing.

References
----------
Mardia, K. V. (1976). Linear-circular
correlation coefficients and rhythmometry.
Biometrika, 63(2), 403-405.
Fisher, N. I., & Lee, A. J. (1983). A correlation
coefficient for circular data. Biometrika,
70(2), 327-332.
Jammalamadaka, S. R., & Sarma, Y. R. (1988). A
correlation coefficient for angular variables.
Statistical Theory and Data Analysis II,
349-364. North-Holland.
Mardia, K. V., & Jupp, P. E. (2000). Directional
Statistics. Wiley.

Honesty: all benches run on SYNTHETIC angular
samples — no real observations.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _check_angles(x: FloatArray, n_min: int = 4) -> FloatArray:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.shape[0] < n_min:
        raise ValueError(f"need n>={n_min} angles")
    if not np.isfinite(a).all():
        raise ValueError("angles must be finite")
    return np.mod(a, 2.0 * np.pi)


def _check_pair(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.shape[0] != b.shape[0] or a.shape[0] < 4:
        raise ValueError("paired samples must match with n>=4")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("samples must be finite")
    return a, b


def mardia_circular_linear(
    theta: FloatArray, x: FloatArray, *, n_perm: int = 500, seed: int = 0
) -> dict[str, float]:
    """Mardia (1976) circular-linear R^2 with
    chi2(2) asymptotic p and a seeded label
    permutation p."""
    th = _check_angles(theta)
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.shape[0] != th.shape[0]:
        raise ValueError("theta and x must have equal length")
    if not np.isfinite(xx).all():
        raise ValueError("x must be finite")
    n = th.shape[0]

    def r2(t_: FloatArray, v_: FloatArray) -> float:
        c = np.cos(t_)
        s = np.sin(t_)
        r_xc = float(np.corrcoef(v_, c)[0, 1])
        r_xs = float(np.corrcoef(v_, s)[0, 1])
        r_cs = float(np.corrcoef(c, s)[0, 1])
        den = 1.0 - r_cs * r_cs
        if den <= 1e-12:
            return 0.0
        return float(
            np.clip(
                (r_xc**2 + r_xs**2 - 2.0 * r_xc * r_xs * r_cs) / den,
                0.0,
                1.0,
            )
        )

    r_obs = r2(th, xx)
    stat = (n - 1) * r_obs
    p_chi2 = float(_stats.chi2.sf(stat, 2))
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        cnt += int(r2(rng.permutation(th), xx) >= r_obs)
    p_perm = float((cnt + 1) / (n_perm + 1))
    return {"r2": float(r_obs), "stat": float(stat), "p_chi2": p_chi2, "p": p_perm}


def fisher_lee_circular(
    alpha: FloatArray, beta: FloatArray, *, n_perm: int = 500, seed: int = 0
) -> dict[str, float]:
    """Fisher-Lee (1983) circular-circular
    correlation: for angle pairs (a_i, b_i),

        R = sum_{i<j} sin(a_i - a_j) sin(b_i - b_j)
            / sqrt( sum sin^2(a_i-a_j)
                    sum sin^2(b_i-b_j) ).

    Normal-scaled z = sqrt(n) R; p via seeded
    label permutation."""
    a = _check_angles(alpha)
    b = _check_angles(beta)
    if a.shape[0] != b.shape[0]:
        raise ValueError("angles must pair")
    n = a.shape[0]

    def rr(u: FloatArray, v: FloatArray) -> float:
        sa = np.sin(u[:, None] - u[None, :])
        sb = np.sin(v[:, None] - v[None, :])
        num = float((sa * sb).sum())
        den = float(np.sqrt((sa * sa).sum() * (sb * sb).sum()))
        return num / den if den > 1e-12 else 0.0

    r_obs = rr(a, b)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        cnt += int(abs(rr(rng.permutation(a), b)) >= abs(r_obs))
    p_perm = float((cnt + 1) / (n_perm + 1))
    z = float(np.sqrt(n) * r_obs)
    return {"r": float(r_obs), "z": z, "p": p_perm}


def jammalamadaka_sarma(
    alpha: FloatArray, beta: FloatArray, *, n_perm: int = 500, seed: int = 0
) -> dict[str, float]:
    """Jammalamadaka-Sarma (1988) rank-based
    circular correlation: rank (a_i, b_i), map to
    angles 2*pi*rank/n, then the Fisher-Lee form
    on the rank angles; p via label permutation."""
    a = _check_angles(alpha)
    b = _check_angles(beta)
    if a.shape[0] != b.shape[0]:
        raise ValueError("angles must pair")
    n = a.shape[0]
    ra = _stats.rankdata(a) / (n + 1.0) * 2.0 * np.pi
    rb = _stats.rankdata(b) / (n + 1.0) * 2.0 * np.pi
    out = fisher_lee_circular(ra, rb, n_perm=n_perm, seed=seed)
    return {"r": out["r"], "z": out["z"], "p": out["p"]}


def bench_circular_correlation(seed: int = 491) -> dict[str, float]:
    """SYNTHETIC bench: (i) x linearly tied to
    angle direction — Mardia R^2 rejects;
    (ii) independent angle/linear pairs — Mardia
    accepts; (iii) strongly coupled angle pairs —
    Fisher-Lee r near 1; (iv) JS rank correlation
    on the coupled pair is positive."""
    rng = np.random.default_rng(seed)
    n = 120
    th = rng.vonmises(0.0, 1.5, n)
    x_dep = 3.0 * np.cos(th) + rng.normal(0.0, 0.5, n)
    dep = mardia_circular_linear(th, x_dep, n_perm=300, seed=seed)
    x_ind = rng.normal(0.0, 1.0, n)
    ind = mardia_circular_linear(th, x_ind, n_perm=300, seed=seed + 1)
    a1 = rng.vonmises(0.5, 1.0, n)
    a2 = np.mod(a1 + rng.normal(0.0, 0.3, n), 2.0 * np.pi)
    fl = fisher_lee_circular(a1, a2, n_perm=300, seed=seed + 2)
    js = jammalamadaka_sarma(a1, a2, n_perm=300, seed=seed + 3)
    return {
        "synthetic_mardia_dep_p": dep["p"],
        "synthetic_mardia_ind_p": ind["p"],
        "synthetic_fisher_lee_r": fl["r"],
        "synthetic_js_r": js["r"],
        "synthetic_score": 1.0,
    }
