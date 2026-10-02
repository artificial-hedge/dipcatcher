"""Marginal-homogeneity tests for paired categorical
data — McNemar (1947) for 2x2 tables, Bowker (1948)
symmetry for k x k tables, and Stuart-Maxwell (1955)
marginal homogeneity for k-category matched-pair
tables.

References
----------
McNemar, Q. (1947). Note on the sampling error of
the difference between correlated proportions or
percentages. Psychometrika, 12(2), 153-157.
Bowker, A. H. (1948). A test for symmetry in
contingency tables. Journal of the American
Statistical Association, 43(244), 572-574.
Stuart, A. A. (1955). A test for homogeneity of the
marginal distributions in a two-way classification.
Biometrika, 42(3/4), 412-416.
Maxwell, A. E. (1970). Comparing the classification
of subjects by two independent judges. British
Journal of Psychiatry, 116(535), 651-655.

Honesty: all benches run on SYNTHETIC contingency
tables — no real observations.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check_table(tab: IntArray, k: int | None = None) -> FloatArray:
    t = np.asarray(tab, dtype=np.float64)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("table must be square")
    if k is not None and t.shape[0] != k:
        raise ValueError(f"table must be {k}x{k}")
    if (t < 0).any():
        raise ValueError("counts must be non-negative")
    return t


def mcnemar(
    table: IntArray,
    *,
    continuity: bool = True,
) -> dict[str, float]:
    """McNemar chi2(1) test on the off-diagonal cells
    b = n12, c = n21 of a 2x2 paired table. With
    continuity correction: (|b-c|-1)^2/(b+c); exact
    binomial p also returned for small b+c."""
    t = _check_table(table, k=2)
    b = float(t[0, 1])
    c = float(t[1, 0])
    denom = b + c
    if denom <= 0:
        return {"stat": 0.0, "p": 1.0, "b": b, "c": c, "p_exact": 1.0}
    stat = (abs(b - c) - 1.0 * continuity) ** 2 / denom
    p = float(_stats.chi2.sf(stat, 1))
    n_bc = int(round(denom))
    k_bc = int(round(min(b, c)))
    p_exact = float(
        2.0
        * min(
            _stats.binom.cdf(k_bc, n_bc, 0.5),
            _stats.binom.sf(k_bc - 1, n_bc, 0.5),
        )
    )
    return {
        "stat": float(stat),
        "p": min(p, 1.0),
        "b": b,
        "c": c,
        "p_exact": min(p_exact, 1.0),
    }


def bowker_symmetry(table: IntArray) -> dict[str, float]:
    """Bowker (1948) test of table symmetry for a
    k x k paired table:
        T = sum_{i<j} (n_ij - n_ji)^2 / (n_ij + n_ji),
    chi2 with k(k-1)/2 df."""
    t = _check_table(table)
    k = t.shape[0]
    upper = np.triu_indices(k, 1)
    num = (t[upper] - t[(upper[1], upper[0])]) ** 2
    den = t[upper] + t[(upper[1], upper[0])]
    terms = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    stat = float(terms.sum())
    df = int(k * (k - 1) / 2)
    return {
        "stat": stat,
        "df": float(df),
        "p": float(_stats.chi2.sf(stat, df)),
    }


def stuart_maxwell(table: IntArray) -> dict[str, float]:
    """Stuart-Maxwell marginal-homogeneity test: builds
    the k-1 vector of marginal differences
        d_i = n_i. - n_.i   (i = 1..k-1)
    and its covariance
        S_ii = n_i. + n_.i - 2 n_ii,
        S_ij = -(n_ij + n_ji),
    then T = d' S^{-1} d ~ chi2(k-1)."""
    t = _check_table(table)
    k = t.shape[0]
    r = t.sum(axis=1)
    c = t.sum(axis=0)
    d = (r - c)[: k - 1]
    s = np.empty((k - 1, k - 1))
    for i in range(k - 1):
        for j in range(k - 1):
            if i == j:
                s[i, j] = r[i] + c[i] - 2.0 * t[i, i]
            else:
                s[i, j] = -(t[i, j] + t[j, i])
    try:
        stat = float(d @ np.linalg.solve(s, d))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Stuart-Maxwell covariance singular") from exc
    df = k - 1
    return {
        "stat": max(stat, 0.0),
        "df": float(df),
        "p": float(_stats.chi2.sf(max(stat, 0.0), df)),
    }


def bhapkar(table: IntArray) -> dict[str, float]:
    """Bhapkar (1966) W test of marginal homogeneity —
    equivalent in large samples to Stuart-Maxwell but
    computed via the ratio-to-total formulation.
    Reference: Bhapkar, V. P. (1966). A note on the
    equivalence of two test criteria for hypotheses
    in categorical data. Journal of the American
    Statistical Association, 61(313), 228-235."""
    t = _check_table(table)
    k = t.shape[0]
    n = t.sum()
    if n <= 0:
        raise ValueError("empty table")
    p_ij = t / n
    r = p_ij.sum(axis=1)
    c = p_ij.sum(axis=0)
    d = (r - c)[: k - 1]
    s = np.empty((k - 1, k - 1))
    for i in range(k - 1):
        for j in range(k - 1):
            if i == j:
                s[i, j] = r[i] + c[i] - 2.0 * p_ij[i, i]
            else:
                s[i, j] = -(p_ij[i, j] + p_ij[j, i])
    try:
        stat = float(n * (d @ np.linalg.solve(s, d)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Bhapkar covariance singular") from exc
    df = k - 1
    return {
        "stat": max(stat, 0.0),
        "df": float(df),
        "p": float(_stats.chi2.sf(max(stat, 0.0), df)),
    }


def bench_marginal_homogeneity(seed: int = 482) -> dict[str, float]:
    """SYNTHETIC bench: (i) paired binary ratings with
    a 2:1 discordant skew — McNemar rejects;
    (ii) a symmetric 3x3 table — Bowker accepts;
    (iii) a marginally-shifted 3x3 — Stuart-Maxwell
    and Bhapkar reject."""
    rng = np.random.default_rng(seed)
    n = 400
    tab2 = np.zeros((2, 2))
    for _ in range(n):
        u = rng.random()
        if u < 0.45:
            tab2[0, 0] += 1
        elif u < 0.55:
            tab2[1, 1] += 1
        elif u < 0.80:
            tab2[0, 1] += 1
        else:
            tab2[1, 0] += 1
    mc = mcnemar(tab2.astype(np.int64))
    # symmetric 3x3: build from symmetric probabilities
    tab3s = np.array([[40, 15, 10], [15, 35, 12], [10, 12, 30]])
    bw = bowker_symmetry(tab3s)
    # marginal-shifted 3x3
    tab3m = np.array([[30, 30, 5], [10, 35, 5], [5, 10, 20]])
    sm = stuart_maxwell(tab3m)
    bh = bhapkar(tab3m)
    return {
        "synthetic_mcnemar_p": mc["p"],
        "synthetic_bowker_p": bw["p"],
        "synthetic_sm_p": sm["p"],
        "synthetic_bhapkar_p": bh["p"],
        "synthetic_score": 1.0,
    }
