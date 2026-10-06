"""Sample uniqueness and the sequential bootstrap (AFML ch. 4).

When labels overlap in time (multi-horizon labels on the same series), the
effective number of independent observations is smaller than the sample
count. This module quantifies that overlap and turns it into:

- ``label_overlap_matrix`` — the binary concurrency matrix of label spans;
- ``avg_uniqueness`` — each sample's average 1/(concurrency) over its life;
- ``sequential_bootstrap`` — a bootstrap that resamples *conditioned on the
  overlaps already drawn*, preserving the dependence structure instead of
  destroying it;
- ``uniqueness_weights`` — sampling weights ∝ uniqueness for bagging.

Honesty: overlap and uniqueness are properties of the label schedule the
caller supplies; weights correct the effective sample size, they do not add
information.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 4 — uniqueness, sequential bootstrap, return attribution.
- Bailey, D. H., López de Prado, M. (2012). The Sharpe ratio efficient
  frontier — effective number of independent bets (spirit).

Composition: pure numpy; deterministic ``np.random.default_rng``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def label_overlap_matrix(t0: IntArray, t1: IntArray) -> BoolArray:
    """Boolean matrix M[i, j] = label i and label j overlap in time.

    ``t0``/``t1`` are integer start (inclusive) and end (exclusive) indices.
    A label always overlaps itself.
    """
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    if t0.shape != t1.shape or t0.ndim != 1:
        raise ValueError("t0 and t1 must be one-dimensional arrays of equal shape")
    if len(t0) == 0:
        return np.zeros((0, 0), dtype=np.bool_)
    if np.any(t1 < t0):
        raise ValueError("each label needs t1 >= t0")
    n = len(t0)
    m = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        # Intersection non-empty ⇔ starts before the other's end and ends
        # after the other's start. Half-open spans: touching ends do not
        # overlap.
        m[i, :] = (t0 < t1[i]) & (t1 > t0[i])
    return m


def avg_uniqueness(t0: IntArray, t1: IntArray) -> FloatArray:
    """Average uniqueness per label: mean over its life of 1/concurrency.

    Concurrency c_t counts labels alive at integer time t. Uniqueness is in
    (0, 1]; disjoint labels have uniqueness exactly 1.
    """
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    if t0.shape != t1.shape or t0.ndim != 1:
        raise ValueError("t0 and t1 must be one-dimensional arrays of equal shape")
    if len(t0) == 0:
        return np.zeros(0, dtype=np.float64)
    if np.any(t1 < t0):
        raise ValueError("each label needs t1 >= t0")
    horizon = int(np.max(t1))
    conc = np.zeros(max(horizon, 1), dtype=np.float64)
    for i in range(len(t0)):
        conc[t0[i] : t1[i]] += 1.0
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1e-12), 0.0)
    out = np.empty(len(t0), dtype=np.float64)
    for i in range(len(t0)):
        life = inv[t0[i] : t1[i]]
        out[i] = float(np.mean(life)) if len(life) > 0 else 0.0
    return out


def sequential_bootstrap(overlap: BoolArray, n_draws: int, *, seed: int = 0) -> IntArray:
    """Sequential bootstrap over labels given their overlap matrix.

    At each step, a candidate is drawn with probability inversely
    proportional to how heavily its overlapping neighbours have already been
    drawn. This preserves the time-series dependence structure of the
    original sample, unlike an i.i.d. bootstrap.
    """
    m = np.asarray(overlap, dtype=np.bool_)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("overlap must be a square matrix")
    n = m.shape[0]
    if n == 0:
        raise ValueError("overlap matrix must be non-empty")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    rng = np.random.default_rng(seed)
    phi = np.ones(n, dtype=np.float64)  # count of draws per label so far
    draws = np.empty(n_draws, dtype=np.int64)
    for step in range(n_draws):
        weighted = (m * phi[np.newaxis, :]).sum(axis=1)
        weights = 1.0 / np.maximum(weighted, 1e-12)
        probs = weights / weights.sum()
        choice = int(rng.choice(n, p=probs))
        draws[step] = choice
        phi[choice] += 1.0
    return draws


def uniqueness_weights(avg_uniq: FloatArray, y: FloatArray | None = None) -> FloatArray:
    """Sampling weights ∝ uniqueness (optionally × |return attribution|).

    Weights are normalised to mean 1, so they can be used directly in
    weighted estimators. With all uniqueness equal the weights are uniform.
    """
    u = np.asarray(avg_uniq, dtype=np.float64)
    if u.ndim != 1 or len(u) == 0:
        raise ValueError("avg_uniq must be a non-empty one-dimensional array")
    if np.any(u < 0):
        raise ValueError("uniqueness must be non-negative")
    w = u.copy()
    if y is not None:
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.shape != u.shape:
            raise ValueError("y must have the same shape as avg_uniq")
        w = w * np.abs(y_arr)
    total = float(w.sum())
    if total <= 0:
        return np.ones(len(u), dtype=np.float64)
    return np.asarray(w * (len(u) / total), dtype=np.float64)
