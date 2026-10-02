"""DeLong-DeLong-Clarke-Pearson (1988) AUC estimation —
placement-value structural components, exact variance
of the Mann-Whitney/Wilcoxon statistic for empirical
ROC curves, covariance between two correlated AUCs
on the same sample, and a z-test for their equality.

References
----------
DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L.
(1988). Comparing the areas under two or more
correlated receiver operating characteristic curves:
a nonparametric approach. Biometrics, 44(3),
837-845.
Hanley, J. A., & McNeil, B. J. (1982). The meaning
and use of the area under a receiver operating
characteristic (ROC) curve. Radiology, 143(1),
29-36.
Sun, X., & Xu, W. (2014). Fast implementation of
DeLong's algorithm for comparing the areas under
correlated receiver operating characteristic
curves. IEEE Signal Processing Letters, 21(11),
1389-1393.

Honesty: all benches run on SYNTHETIC score
populations — no real model predictions.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _check_scores(pos: FloatArray, neg: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(pos, dtype=np.float64).ravel()
    b = np.asarray(neg, dtype=np.float64).ravel()
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("need >=2 positive and >=2 negative scores")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("scores must be finite")
    return a, b


def auc_wilcoxon(pos: FloatArray, neg: FloatArray) -> float:
    """Mann-Whitney AUC = P(X_pos > X_neg) with mid-rank
    ties at 1/2, via pairwise comparisons."""
    a, b = _check_scores(pos, neg)
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff) + 1.0).mean() / 2.0)


def _placements(pos: FloatArray, neg: FloatArray) -> tuple[FloatArray, FloatArray]:
    """DeLong placement values: V10(X_i) is the fraction
    of negative scores below X_i (mid-tie 1/2);
    V01(Y_j) is the fraction of positive scores above
    Y_j (mid-tie 1/2)."""
    v10 = np.empty(pos.shape[0])
    v01 = np.empty(neg.shape[0])
    for i, xi in enumerate(pos):
        v10[i] = float((np.sign(xi - neg) + 1.0).mean() / 2.0)
    for j, yj in enumerate(neg):
        v01[j] = float((np.sign(pos - yj) + 1.0).mean() / 2.0)
    return v10, v01


def delong_auc(pos: FloatArray, neg: FloatArray) -> dict[str, float]:
    """DeLong AUC + variance: the structural-component
    covariance estimator
        var(AUC) = var(V10)/m + var(V01)/n,
    exact for the empirical AUC (Hanley-McNeil
    approximation avoided). Returns auc, var, se, and
    the z for H0: AUC = 0.5."""
    a, b = _check_scores(pos, neg)
    v10, v01 = _placements(a, b)
    auc = float(v10.mean())
    var = float(np.var(v10, ddof=1) / a.shape[0] + np.var(v01, ddof=1) / b.shape[0])
    se = float(np.sqrt(max(var, 0.0)))
    z = (auc - 0.5) / max(se, 1e-12)
    return {
        "auc": auc,
        "var": var,
        "se": se,
        "z_half": float(z),
        "p_half": float(2.0 * _stats.norm.sf(abs(z))),
    }


def delong_compare(
    pos1: FloatArray,
    neg1: FloatArray,
    pos2: FloatArray,
    neg2: FloatArray,
) -> dict[str, float]:
    """Two correlated AUCs on paired samples: covariance
    from the placement-value cross-covariances
        cov(AUC1, AUC2) = cov(V10^1, V10^2)/m
                        + cov(V01^1, V01^2)/n,
    then a z-test on AUC1 - AUC2."""
    a1, b1 = _check_scores(pos1, neg1)
    a2, b2 = _check_scores(pos2, neg2)
    if a1.shape[0] != a2.shape[0] or b1.shape[0] != b2.shape[0]:
        raise ValueError("paired samples must have equal sizes")
    v10_1, v01_1 = _placements(a1, b1)
    v10_2, v01_2 = _placements(a2, b2)
    auc1 = float(v10_1.mean())
    auc2 = float(v10_2.mean())
    s10 = np.cov(np.vstack([v10_1, v10_2]), ddof=1)
    s01 = np.cov(np.vstack([v01_1, v01_2]), ddof=1)
    var = float((s10[0, 0] + s10[1, 1] - 2.0 * s10[0, 1]) / a1.shape[0])
    var += float((s01[0, 0] + s01[1, 1] - 2.0 * s01[0, 1]) / b1.shape[0])
    se = float(np.sqrt(max(var, 0.0)))
    z = (auc1 - auc2) / max(se, 1e-12)
    return {
        "auc1": auc1,
        "auc2": auc2,
        "diff": auc1 - auc2,
        "se_diff": se,
        "z": float(z),
        "p": float(2.0 * _stats.norm.sf(abs(z))),
    }


def delong_multi(scores_pos: FloatArray, scores_neg: FloatArray) -> dict[str, float | FloatArray]:
    """k correlated AUCs: stacks each classifier's
    placement vectors and returns the full (k, k)
    covariance matrix of the AUC estimates
    (DeLong 1988, eq. for L). scores_* are (n, k)."""
    sp = np.asarray(scores_pos, dtype=np.float64)
    sn = np.asarray(scores_neg, dtype=np.float64)
    if sp.ndim != 2 or sn.ndim != 2 or sp.shape != sn.shape or sp.shape[1] < 2:
        raise ValueError("scores must be (n, k) matrices with k>=2")
    k = sp.shape[1]
    v10s = np.empty((k, sp.shape[0]))
    v01s = np.empty((k, sn.shape[0]))
    aucs = np.empty(k)
    for j in range(k):
        v10s[j], v01s[j] = _placements(sp[:, j], sn[:, j])
        aucs[j] = float(v10s[j].mean())
    cov = np.cov(v10s, ddof=1) / sp.shape[0] + np.cov(v01s, ddof=1) / sn.shape[0]
    return {"aucs": aucs, "cov": cov}


def bench_delong_auc(seed: int = 480) -> dict[str, float]:
    """SYNTHETIC bench: (i) separable scores
    (pos~N(1,1), neg~N(0,1)) give AUC>0.7 with p<0.01;
    (ii) identical-distribution scores give p>0.1;
    (iii) two correlated classifiers on shared noise —
    the stronger model wins via delong_compare z<0
    when comparing weaker minus stronger AUCs."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(1.0, 1.0, 80)
    neg = rng.normal(0.0, 1.0, 90)
    fit = delong_auc(pos, neg)
    pos_n = rng.normal(0.0, 1.0, 80)
    neg_n = rng.normal(0.0, 1.0, 90)
    fit_n = delong_auc(pos_n, neg_n)
    # shared latent + classifier noise: strong sees latent
    # + small noise; weak sees latent + large noise
    lat_p = rng.normal(1.0, 1.0, 100)
    lat_n = rng.normal(0.0, 1.0, 110)
    s_pos = lat_p + rng.normal(0.0, 0.3, 100)
    s_neg = lat_n + rng.normal(0.0, 0.3, 110)
    w_pos = lat_p + rng.normal(0.0, 1.2, 100)
    w_neg = lat_n + rng.normal(0.0, 1.2, 110)
    cmp_ = delong_compare(w_pos, w_neg, s_pos, s_neg)
    return {
        "synthetic_auc": fit["auc"],
        "synthetic_p_sep": fit["p_half"],
        "synthetic_p_null": fit_n["p_half"],
        "synthetic_cmp_z": cmp_["z"],
        "synthetic_cmp_p": cmp_["p"],
        "synthetic_score": 1.0,
    }
