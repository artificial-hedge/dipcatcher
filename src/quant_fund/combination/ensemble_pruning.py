"""Greedy ensemble pruning with replacement (Caruana-style).

Selecting a subset of members often beats combining all of them — one
bad member can drag the ensemble. This module implements greedy forward
selection with replacement on a validation loss:

- ``greedy_prune`` — start empty; repeatedly add the member (with
  replacement) that minimises the running combined loss; stop after
  ``max_steps``; the multiset counts define the pruned weights;
- ``pruned_weights`` — normalise the selection counts into simplex
  weights;
- ``pruning_gain`` — validation loss of the pruned ensemble versus the
  full equal-weight ensemble.

Honesty: pruning is fitted on the supplied validation window; the gain is
in-window, not a guarantee.

References:
- Caruana, R., Niculescu-Mizil, A., Crew, G., Ksikes, A. (2004).
  Ensemble selection from libraries of models — greedy with replacement.
- Zhou, Z.-H. (2012). *Ensemble Methods* — ensemble pruning taxonomy.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _mean_loss(q: FloatArray, y: FloatArray, taus: FloatArray) -> float:
    diff = y[..., None] - q
    loss = np.maximum(taus * diff, (taus - 1.0) * diff)
    return float(np.mean(loss))


def greedy_prune(
    member_quantiles: FloatArray,
    y: FloatArray,
    taus: FloatArray,
    *,
    max_steps: int = 30,
) -> dict[str, np.ndarray | int | float]:
    """Greedy forward selection with replacement on mean pinball.

    ``member_quantiles`` is (T, M, Q). Returns the selection counts (M,),
    the number of steps taken, and the in-sample pinball of the pruned
    ensemble.
    """
    q = np.asarray(member_quantiles, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    if q.ndim != 3 or y_arr.shape != (q.shape[0],):
        raise ValueError("member_quantiles must be (T, M, Q) with y (T,)")
    if taus_arr.shape != (q.shape[2],):
        raise ValueError("taus must be (Q,)")
    t_total, m, _ = q.shape
    counts = np.zeros(m, dtype=np.float64)
    chosen: list[int] = []
    for _ in range(max_steps):
        best_k = -1
        best_loss = np.inf
        for k in range(m):
            trial = chosen + [k]
            w = np.bincount(trial, minlength=m).astype(np.float64)
            w = w / w.sum()
            combined = np.einsum("m,tmq->tq", w, q)
            loss = _mean_loss(combined, y_arr, taus_arr)
            if loss < best_loss:
                best_loss = loss
                best_k = k
        chosen.append(best_k)
        counts[best_k] += 1.0
    w_final = counts / counts.sum()
    combined = np.einsum("m,tmq->tq", w_final, q)
    return {
        "counts": counts,
        "steps": len(chosen),
        "pinball": _mean_loss(combined, y_arr, taus_arr),
    }


def pruned_weights(counts: FloatArray) -> FloatArray:
    """Normalise selection counts to simplex weights (equal fallback)."""
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 1 or len(c) == 0:
        raise ValueError("counts must be a non-empty one-dimensional array")
    total = float(c.sum())
    if total <= 0:
        return np.full(len(c), 1.0 / len(c))
    return np.asarray(c / total, dtype=np.float64)


def pruning_gain(
    member_quantiles: FloatArray,
    y: FloatArray,
    taus: FloatArray,
    *,
    max_steps: int = 30,
) -> dict[str, float]:
    """Validation pinball of the pruned ensemble vs equal weights."""
    q = np.asarray(member_quantiles, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    out = greedy_prune(q, y_arr, taus_arr, max_steps=max_steps)
    counts = np.asarray(out["counts"], dtype=np.float64)
    w = pruned_weights(counts)
    pruned_q = np.einsum("m,tmq->tq", w, q)
    pruned_loss = _mean_loss(pruned_q, y_arr, taus_arr)
    equal_q = np.mean(q, axis=1)
    equal_loss = _mean_loss(equal_q, y_arr, taus_arr)
    return {
        "pruned_pinball": pruned_loss,
        "equal_pinball": equal_loss,
        "n_selected": float(np.sum(counts > 0)),
    }
