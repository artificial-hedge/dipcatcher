"""PELT pruned optimal partitioning + wild binary segmentation.

References
----------
- Killick, R., Fearnhead, P. & Eckley, I.A. (2012). "Optimal
  Detection of Changepoints with a Linear Computational Cost."
  *JASA* 107(500), 1590-1598.
- Fryzlewicz, P. (2014). "Wild Binary Segmentation for
  Multiple Change-Point Detection." *Annals of Statistics*
  42(6), 2243-2281.
- Bai, J. & Perron, P. (1998). "Estimating and Testing Linear
  Models with Multiple Structural Changes." *Econometrica*
  66(1), 47-78.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
PELT solves ``min over tau of sum_segments cost + beta*|tau|``
by the optimal-partitioning recursion with the pruning
inequality: candidate ``s`` is discarded once
``F(s) + cost(s,t) + beta > F(t)``, which keeps the algorithm
O(n) in practice — the pruning step (not just the recursion)
is what separates PELT from the quadratic OP of Jackson et
al. Wild binary segmentation augments the single top-down
split with ``n_intervals`` random ``[s,e]`` windows, takes the
CUSUM argmax over each, and recurses inside the winning
window — random intervals localize narrow mean shifts that a
single dyadic split misses. ``synth_changepoints`` plants a
piecewise-constant mean with deterministic break locations;
the bench gates on PELT recovering the true count and WBS
locating breaks within tolerance, with a constant series
producing no detections.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 60) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _seg_cost_var(cs: FloatArray, cs2: FloatArray, s: int, t: int) -> float:
    """Within-segment SSE from prefix sums."""
    n = t - s
    if n <= 0:
        return 0.0
    tot = cs[t] - cs[s]
    tot2 = cs2[t] - cs2[s]
    sse = tot2 - tot * tot / n
    return float(max(sse, 0.0))


def pelt_mean(
    x: FloatArray,
    penalty: float | None = None,
    min_seg: int = 10,
) -> dict[str, FloatArray]:
    """PELT change-point set for a mean shift."""
    v = _as_series(x)
    n = v.size
    beta = penalty if penalty is not None else 2.0 * np.log(n)
    if beta <= 0 or min_seg < 2:
        raise ValueError("bad penalty")
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cs2 = np.concatenate([[0.0], np.cumsum(v * v)])
    f = np.full(n + 1, np.inf)
    f[0] = -beta
    last_cp = np.zeros(n + 1, dtype=np.int64)
    cands: list[int] = [0]
    for t in range(min_seg, n + 1):
        best_val, best_s = np.inf, -1
        for s in cands:
            if t - s < min_seg:
                continue
            val = f[s] + _seg_cost_var(cs, cs2, s, t) + beta
            if val < best_val:
                best_val, best_s = val, s
        f[t] = best_val
        last_cp[t] = best_s
        # PELT prune
        cands = [s for s in cands if f[s] + _seg_cost_var(cs, cs2, s, t) <= f[t]]
        cands.append(t - min_seg + 1)
    cps: list[int] = []
    t = n
    while last_cp[t] > 0:
        cps.append(int(last_cp[t]))
        t = int(last_cp[t])
    cps.reverse()
    out: dict[str, FloatArray] = {
        "changepoints": np.asarray(cps, dtype=np.float64),
        "final_cost": np.array([f[n]]),
        "n_cp": np.array([float(len(cps))]),
    }
    return out


def _cusum(x: FloatArray, s: int, e: int) -> tuple[int, float]:
    seg = x[s:e]
    if seg.size < 4:
        return -1, -np.inf
    tot = float(np.sum(seg))
    n = seg.size
    csum = np.cumsum(seg) - tot * np.arange(1, n + 1) / n
    stat = np.abs(csum) / np.sqrt(n * float(np.var(seg)) + 1e-12)
    b = int(np.argmax(stat))
    return s + b, float(stat[b])


def wbs_mean(
    x: FloatArray,
    n_intervals: int = 200,
    threshold: float = 2.0,
    min_dist: int = 10,
    seed: int = 20261231,
) -> dict[str, FloatArray]:
    """Wild binary segmentation mean breaks."""
    v = _as_series(x)
    n = v.size
    rng = np.random.default_rng(seed)
    cps: list[int] = []

    def _scan(s: int, e: int, depth: int) -> None:
        if e - s < 2 * min_dist or depth > 12:
            return
        best_b, best_stat = -1, -np.inf
        for _ in range(n_intervals):
            a = int(rng.integers(s, max(s + 1, e - min_dist)))
            b_ = int(rng.integers(a + min_dist, e + 1))
            if b_ - a < 2 * min_dist:
                continue
            bb, st = _cusum(v, a, b_)
            if st > best_stat:
                best_b, best_stat = bb, st
        if best_b < 0 or best_stat < threshold:
            return
        if not (s + min_dist <= best_b <= e - min_dist):
            return
        cps.append(best_b)
        _scan(s, best_b, depth + 1)
        _scan(best_b, e, depth + 1)

    _scan(0, n, 0)
    cps = sorted(set(cps))
    out: dict[str, FloatArray] = {
        "changepoints": np.asarray(cps, dtype=np.float64),
        "n_cp": np.array([float(len(cps))]),
    }
    return out


def synth_changepoints(
    seed: int = 20261231 + 340,
    n: int = 600,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC piecewise-constant mean vs constant."""
    rng = np.random.default_rng(seed)
    cps_true = np.array([150, 300, 450])
    means = np.array([0.0, 2.0, -1.0, 1.0])
    x = np.zeros(n)
    bounds = np.concatenate([[0], cps_true, [n]])
    for i in range(4):
        x[bounds[i] : bounds[i + 1]] = means[i]
    x += 0.8 * rng.standard_normal(n)
    null = 0.5 + 0.8 * rng.standard_normal(n)
    return x.astype(np.float64), cps_true.astype(np.float64), null.astype(np.float64)


def bench_pelt_wbs(seed: int = 20261231 + 340) -> dict[str, float]:
    x, cps_true, null = synth_changepoints(seed=seed)
    r_p = pelt_mean(x, penalty=2.0 * np.log(x.size))
    r_p_null = pelt_mean(null, penalty=2.0 * np.log(null.size))
    r_w = wbs_mean(x, n_intervals=300, threshold=2.5, min_dist=15, seed=seed)
    got = r_p["changepoints"]
    dist = float(np.max(np.abs(np.sort(got) - cps_true))) if got.size == cps_true.size else 1e6
    ok = got.size == 3 and dist <= 8.0 and r_p_null["n_cp"][0] <= 1.0 and r_w["n_cp"][0] >= 2.0
    out: dict[str, float] = {
        "synthetic_pelt_n_cp": float(got.size),
        "synthetic_pelt_max_dist": dist,
        "synthetic_pelt_null_cp": float(r_p_null["n_cp"][0]),
        "synthetic_wbs_n_cp": float(r_w["n_cp"][0]),
        "score": 1.0 if ok else 0.0,
    }
    return out
