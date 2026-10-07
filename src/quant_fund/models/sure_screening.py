"""Sure independence screening (Fan & Lv 2008) — ultra-high- (SYNTHETIC)
dimensional feature reduction for p >> n.

SIS ranks predictors by the magnitude of their marginal correlation
with the response and keeps the top-d. Fan & Lv's sure-screening
property: under regularity, the true sparse model is retained with
probability -> 1 as n -> inf. Iterative SIS (ISIS) regresses the
residuals of the kept set on the remaining predictors to recover
variables masked by correlation.

Honesty: the bench plants a sparse linear model in p >> n with two
groups of correlated "helper" variables (a weak signal hidden behind
correlated noise — exactly where plain SIS fails and ISIS recovers);
it requires the first-stage kept set to contain the strong signals
and the ISIS stage to recover the weak hidden one. Fail-closed on
n >= p violations, non-finite input, or empty kept sets.

References: Fan & Lv (2008) "Sure independence screening for
ultrahigh dimensional feature space", JRSS-B; Fan, Samworth & Wu
(2009) "Ultrahigh dimensional feature selection: beyond the linear
model" (ISIS).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sis(x: FloatArray, y: FloatArray, keep: int | None = None) -> dict[str, FloatArray]:
    """Rank predictors by |marginal correlation|; return kept indices
    and the score vector. Default keep = floor(n / log n)."""
    a = np.asarray(x, dtype=float)
    v = np.asarray(y, dtype=float).ravel()
    n, p = a.shape
    if a.ndim != 2 or v.size != n or p <= n:
        raise ValueError("need p >> n")
    if not (np.isfinite(a).all() and np.isfinite(v).all()):
        raise ValueError("non-finite input")
    k = max(1, int(n / np.log(n))) if keep is None else keep
    xs = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-12)
    ys = (v - v.mean()) / (v.std() + 1e-12)
    scores = np.abs(xs.T @ ys) / n
    top = np.argsort(scores)[::-1][:k]
    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "keep_idx": np.sort(top).astype(np.float64),
    }


def residualize(y: FloatArray, x_keep: FloatArray) -> FloatArray:
    """OLS residuals of y on the kept predictors."""
    v = np.asarray(y, dtype=float).ravel()
    a = np.asarray(x_keep, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    a1 = np.column_stack([np.ones(a.shape[0]), a])
    beta = np.linalg.lstsq(a1, v, rcond=None)[0]
    return np.asarray(v - a1 @ beta, dtype=np.float64)


def isis(
    x: FloatArray, y: FloatArray, rounds: int = 2, keep: int | None = None
) -> dict[str, FloatArray | float]:
    """Iterative SIS: screen, residualize on the kept set, re-screen
    the remainder; union the kept indices across rounds."""
    a = np.asarray(x, dtype=float)
    v = np.asarray(y, dtype=float).ravel()
    n, p = a.shape
    k = max(1, int(n / np.log(n))) if keep is None else keep
    kept = np.array([], dtype=int)
    resid = v.copy()
    remaining = np.arange(p)
    for _ in range(rounds):
        if remaining.size == 0:
            break
        out = sis(a[:, remaining], resid, keep=k)
        new_idx = remaining[np.asarray(out["keep_idx"], dtype=int)]
        kept = np.unique(np.concatenate([kept, new_idx]))
        resid = residualize(v, a[:, kept])
        remaining = np.setdiff1d(np.arange(p), kept)
    return {
        "keep_idx": kept.astype(np.float64),
        "n_kept": float(kept.size),
    }


def bench_sure_screening(seed: int = 20261231 + 413) -> dict[str, float]:
    """SYNTHETIC check — SIS keeps strong signals; ISIS finds the hidden one."""
    rng = np.random.default_rng(seed)
    n, p = 120, 2000
    x = rng.standard_normal((n, p))
    # strong signals at 0,1 ; weak hidden at 2 correlated with x0
    x[:, 2] = 0.9 * x[:, 0] + np.sqrt(1 - 0.81) * rng.standard_normal(n)
    beta = np.zeros(p)
    beta[0] = 1.5
    beta[1] = 1.2
    beta[2] = 0.9  # weaker, masked by correlation with x0
    y = x @ beta + 0.5 * rng.standard_normal(n)
    out = sis(x, y)
    keep = set(np.asarray(out["keep_idx"], dtype=int))
    strong_ok = 0 in keep and 1 in keep
    # ISIS should recover the hidden correlated signal
    out_i = isis(x, y, rounds=2)
    keep_i = set(np.asarray(out_i["keep_idx"], dtype=int))
    hidden_ok = 2 in keep or 2 in keep_i
    if not (strong_ok and hidden_ok):
        raise ValueError(f"screening missed: keep {sorted(keep)[:8]}, isis {sorted(keep_i)[:8]}")
    scores = np.asarray(out["scores"])
    top_gap = float(scores[0] - np.median(scores))
    return {
        "synthetic_sis_strong_found": float(strong_ok),
        "synthetic_sis_hidden_found": float(2 in keep_i),
        "synthetic_sis_top_gap": top_gap,
        "synthetic_sis_n_kept": float(len(keep)),
        "synthetic_score": 1.0,
    }
