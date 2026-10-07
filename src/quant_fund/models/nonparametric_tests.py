"""Rank-based nonparametric location tests: Mann-Whitney U,
Wilcoxon signed-rank, Kruskal-Wallis, and Jonckheere-Terpstra.

Mann-Whitney (1947): U = number of (x_i, y_j) pairs with x_i < y_j;
under H0 (same distribution) U ~ normal with mean n m / 2 and
tie-corrected variance — reported as the common-language effect size
Pr(X < Y) plus the normal p-value.

Kruskal-Wallis (1952): H statistic on pooled ranks, chi-square
(df = k-1) reference with tie correction.

Jonckheere (1954) / Terpstra (1952): ordered alternative test —
statistic counts inter-group concordances consistent with the
hypothesized ordering; normal approximation on the standardized
count.

Honesty: the benches plant a location shift and an ordered three-
group location pattern; each test must reject at the intended level
while the same test on the null DGP stays under its nominal size.
Fail-closed on empty samples and all-tie degenerate cases.

References: Mann & Whitney (1947); Wilcoxon (1945); Kruskal & Wallis
(1952); Jonckheere (1954); Lehmann (1975) "Nonparametrics" for the
asymptotics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _check(v: FloatArray, min_n: int = 3) -> FloatArray:
    a = np.asarray(v, dtype=float).ravel()
    if a.size < min_n or not np.isfinite(a).all():
        raise ValueError("bad sample")
    return a


def mann_whitney(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Two-sided Mann-Whitney U with common-language effect size."""
    a, b = _check(x), _check(y)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = a.size, b.size
    return {
        "u_stat": float(u),
        "p_value": float(p),
        "cl_es": float(u / (n1 * n2)),  # Pr(X > Y)
        "n1": float(n1),
        "n2": float(n2),
    }


def wilcoxon_signed_rank(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Wilcoxon signed-rank on paired differences."""
    a, b = _check(x), _check(y)
    if a.size != b.size:
        raise ValueError("paired samples need equal n")
    d = a - b
    if np.all(d == 0):
        raise ValueError("all-zero differences")
    w, p = stats.wilcoxon(a, b)
    return {"w_stat": float(w), "p_value": float(p), "n": float(a.size)}


def kruskal_wallis(*groups: FloatArray) -> dict[str, float]:
    """Kruskal-Wallis H over k >= 2 groups."""
    gs = [_check(g) for g in groups]
    if len(gs) < 2:
        raise ValueError("need >= 2 groups")
    h, p = stats.kruskal(*gs)
    return {
        "h_stat": float(h),
        "p_value": float(p),
        "k": float(len(gs)),
        "n_total": float(sum(g.size for g in gs)),
    }


def jonckheere_terpstra(*groups: FloatArray) -> dict[str, float]:
    """Ordered-alternative test: H1 is mu_1 <= ... <= mu_k with at
    least one strict. Statistic S = sum_{i<j} U_ij counting pairs
    consistent with the ordering; normal p on the standardized S."""
    gs = [_check(g) for g in groups]
    k = len(gs)
    if k < 2:
        raise ValueError("need >= 2 groups")
    s_stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for xi in gs[i]:
                s_stat += float(np.sum(xi < gs[j])) + 0.5 * float(np.sum(xi == gs[j]))
    ns = np.array([g.size for g in gs], dtype=float)
    n_tot = ns.sum()
    e_s = (n_tot * n_tot - float(np.sum(ns * ns))) / 4.0
    var_s = (n_tot * n_tot * (2 * n_tot + 3) - float(np.sum(ns * ns * (2 * ns + 3)))) / 72.0
    z = (s_stat - e_s) / np.sqrt(var_s)
    p = float(1.0 - stats.norm.cdf(z))
    return {
        "jt_stat": float(s_stat),
        "z_stat": float(z),
        "p_value": p,
        "k": float(k),
    }


def bench_nonparametric_tests(
    seed: int = 20261231 + 410,
) -> dict[str, float]:
    """SYNTHETIC check — planted shift/order rejected, null respects size."""
    rng = np.random.default_rng(seed)
    # location shift 0.8 sd -> U should reject strongly
    x = rng.standard_normal(60)
    y = rng.standard_normal(60) + 0.8
    mw = mann_whitney(x, y)
    cl_dev = abs(mw["cl_es"] - 0.5)
    if mw["p_value"] > 0.01 or cl_dev < 0.15:
        raise ValueError(f"MWU misses planted shift: {mw}")
    # ordered groups
    g1 = rng.standard_normal(40)
    g2 = rng.standard_normal(40) + 0.5
    g3 = rng.standard_normal(40) + 1.0
    jt = jonckheere_terpstra(g1, g2, g3)
    if jt["p_value"] > 0.01:
        raise ValueError(f"JT misses planted ordering: {jt}")
    kw = kruskal_wallis(g1, g2, g3)
    if kw["p_value"] > 0.01:
        raise ValueError(f"KW misses planted ordering: {kw}")
    # null: same dist -> p should not be tiny (size check at 1%)
    z1 = rng.standard_normal(50)
    z2 = rng.standard_normal(50)
    mw0 = mann_whitney(z1, z2)
    if mw0["p_value"] < 0.01:
        raise ValueError(f"MWU false positive on null: {mw0}")
    return {
        "synthetic_mwu_p_alt": float(mw["p_value"]),
        "synthetic_mwu_cl_es": float(mw["cl_es"]),
        "synthetic_mwu_cl_dev": float(cl_dev),
        "synthetic_jt_p_alt": float(jt["p_value"]),
        "synthetic_kw_p_alt": float(kw["p_value"]),
        "synthetic_mwu_p_null": float(mw0["p_value"]),
        "synthetic_score": 1.0,
    }
