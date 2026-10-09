"""Multi-signal IC matrix structure.

Stacks per-signal IC series into a matrix and quantifies the structure of
the signal set:

- ``ic_matrix`` — (T, S) IC series per signal from aligned panels;
- ``signal_ic_correlation`` — correlation matrix of the IC series;
- ``ic_factor_structure`` — eigenvalue concentration of the IC covariance:
  effective number of IC factors (participation ratio) and the top
  eigenvalue share — high concentration means signals share one beta;
- ``signal_clusters`` — simple agglomerative grouping by IC correlation
  (average linkage, correlation distance), returning flat labels.

Honesty: correlation/clustering describe the supplied evaluation window;
they do not establish stable signal taxonomy.

References:
- López de Prado, M. (2020). *Machine Learning for Asset Managers* — the
  IC matrix, eigenvalue clipping, and hierarchical signal clustering.
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective
  function — average-linkage clustering.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.decay.ic_series import spearman_ic

FloatArray = NDArray[np.float64]


def ic_matrix(preds: list[FloatArray], actual: FloatArray) -> FloatArray:
    """(T, S) matrix of per-signal rank IC series from aligned panels."""
    if not preds:
        raise ValueError("preds must be non-empty")
    a = np.asarray(actual, dtype=np.float64)
    cols = []
    for p in preds:
        pa = np.asarray(p, dtype=np.float64)
        if pa.shape != a.shape or pa.ndim != 2:
            raise ValueError("each pred must be (T, N) matching actual")
        cols.append(spearman_ic(pa, a))
    return np.column_stack(cols)


def signal_ic_correlation(ic: FloatArray) -> FloatArray:
    """Correlation matrix of the per-signal IC series."""
    m = np.asarray(ic, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("ic must be (T, S) with S >= 2")
    out = np.corrcoef(m, rowvar=False)
    return np.asarray(np.clip(out, -1.0, 1.0), dtype=np.float64)


def ic_factor_structure(ic: FloatArray) -> dict[str, float]:
    """Eigenvalue concentration of the IC covariance matrix.

    Returns the participation ratio (effective number of factors) and the
    top-eigenvalue share. A single shared factor drives both: PR near 1
    and share near 1.
    """
    m = np.asarray(ic, dtype=np.float64)
    m = m[np.isfinite(m).all(axis=1)]
    if m.shape[0] < 10 or m.shape[1] < 2:
        raise ValueError("need at least 10 valid periods and 2 signals")
    cov = np.cov(m, rowvar=False)
    eig = np.linalg.eigvalsh(cov)[::-1]
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if total <= 0:
        return {"participation_ratio": 0.0, "top_share": 0.0}
    pr = float((total**2) / float(np.sum(eig * eig)))
    return {"participation_ratio": pr, "top_share": float(eig[0] / total)}


def signal_clusters(ic: FloatArray, n_clusters: int = 2) -> np.ndarray:
    """Average-linkage clustering of signals by IC correlation distance.

    Simple agglomerative implementation (O(S³), fine for small signal
    sets): merge the pair with the highest IC correlation until
    ``n_clusters`` groups remain; labels are arbitrary but stable.
    """
    m = np.asarray(ic, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] < n_clusters:
        raise ValueError("ic must be (T, S) with S >= n_clusters")
    corr = signal_ic_correlation(m)
    s = corr.shape[0]
    groups: list[list[int]] = [[i] for i in range(s)]
    while len(groups) > n_clusters:
        best = (-2.0, 0, 1)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                sim = float(np.mean(corr[np.ix_(groups[i], groups[j])]))
                if sim > best[0]:
                    best = (sim, i, j)
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]
    labels = np.empty(s, dtype=np.int64)
    for lab, g in enumerate(groups):
        for idx in g:
            labels[idx] = lab
    return labels
