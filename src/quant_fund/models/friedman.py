"""Friedman, Kendall's W, and Page's L — blocked rank tests (SYNTHETIC).

Friedman (1937): for a blocked design (n blocks x k treatments) the
treatment ranks within each block give

    Q = (12 n / (k(k+1))) [sum_j Rbar_j^2 - k (k+1)^2 / 4]   ~ chi^2_{k-1}

Kendall's W = Q / (n (k-1)) is the coefficient of concordance in
(0,1]. Page (1963) tests a *predicted* ordering j_1<...<j_k via
L = sum_j j R_j ~ N(mu_L, sigma_L^2) with
mu_L = n k (k+1)^2 / 4, sigma_L^2 = n k^2 (k+1) (k^2-1) / 144.

Honesty: the bench plants a monotone treatment effect (all three
tests must reject) and a null (all respect size at loose bounds).
The chi^2/normal approximations are standard; bounds documented.
Fail-closed on non-2D input, ties-only columns, or k<3.

References: Friedman (1937) "The use of ranks to avoid the
assumption of normality"; Kendall & Babington Smith (1939)
concordance; Page (1963) "Ordered hypotheses for multiple
treatments"; Hollander, Wolfe, Chicken (2014) ch. 7.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

FloatArray = NDArray[np.float64]


def _check_blocks(x: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 4 or a.shape[1] < 3 or not np.isfinite(a).all():
        raise ValueError("bad block matrix")
    return a


def _ranks_rows(a: FloatArray) -> FloatArray:
    """Average ranks within each row (block)."""
    from scipy.stats import rankdata

    return np.asarray(np.apply_along_axis(rankdata, 1, a), dtype=np.float64)


def friedman_test(x: FloatArray) -> dict[str, float | FloatArray]:
    """Friedman Q test on an (n blocks, k treatments) matrix."""
    a = _check_blocks(x)
    n, k = a.shape
    if not np.isfinite(a).all() or (a.std(axis=1) <= 1e-12).any():
        raise ValueError("degenerate block")
    r = _ranks_rows(a)
    rbar = r.mean(axis=0)
    q = float(12.0 * n / (k * (k + 1)) * ((rbar**2).sum() - k * (k + 1) ** 2 / 4.0))
    p = float(chi2.sf(max(0.0, q), k - 1))
    w = q / (n * (k - 1)) if n * (k - 1) > 0 else 0.0
    return {
        "q": q,
        "p": p,
        "kendall_w": float(w),
        "mean_ranks": np.asarray(rbar, dtype=np.float64),
    }


def page_l(x: FloatArray) -> dict[str, float]:
    """Page's L for a predicted increasing treatment ordering.

    Tests H_1: column means increase left-to-right. One-sided normal
    approximation.
    """
    a = _check_blocks(x)
    n, k = a.shape
    r = _ranks_rows(a)
    rj = r.sum(axis=0)  # rank sums per treatment
    l_stat = float((np.arange(1, k + 1) * rj).sum())
    mu = n * k * (k + 1) ** 2 / 4.0
    var = n * k * k * (k + 1) * (k * k - 1) / 144.0
    z = (l_stat - mu) / np.sqrt(var)
    p = float(norm.sf(z))
    return {"l": l_stat, "z": float(z), "p": p}


def bench_friedman(seed: int = 20261231 + 429) -> dict[str, float]:
    """SYNTHETIC check — ordered effect rejected by all three, null held."""
    rng = np.random.default_rng(seed)
    n, k = 25, 4
    base = rng.standard_normal((n, 1))
    x = base + np.array([0.0, 0.4, 0.9, 1.4])[None, :] + 0.4 * rng.standard_normal((n, k))
    f = friedman_test(x)
    pg = page_l(x)
    xn = base + 0.4 * rng.standard_normal((n, k))
    fn = friedman_test(xn)
    pgn = page_l(xn)
    if float(f["p"]) > 0.01 or pg["p"] > 0.01 or float(fn["p"]) < 0.005 or pgn["p"] < 0.005:
        raise ValueError(
            f"friedman off: f={float(f['p']):.4f} page={pg['p']:.4f} "
            f"null_f={float(fn['p']):.4f} null_p={pgn['p']:.4f}"
        )
    return {
        "synthetic_friedman_p": float(f["p"]),
        "synthetic_friedman_w": float(f["kendall_w"]),
        "synthetic_page_p": float(pg["p"]),
        "synthetic_friedman_p_null": float(fn["p"]),
        "synthetic_score": 1.0,
    }
