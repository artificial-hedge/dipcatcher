"""Sakoe-Chiba dynamic time warping + curve registration.

References
----------
- Sakoe, H. & Chiba, S. (1978). "Dynamic Programming
  Algorithm Optimization for Spoken Word Recognition."
  *IEEE Trans. ASSP* 26(1), 43-49.
- Berndt, D.J. & Clifford, J. (1994). "Using Dynamic Time
  Warping to Find Patterns in Time Series." *KDD Workshop*
  10(16), 359-370.
- Marron, J.S., Ramsay, J.O., Sangalli, L.M. & Srivastava,
  A. (2015). "Functional Data Analysis of Amplitude and
  Phase Variation." *Statistical Science* 30(4), 468-484.
- Srivastava, A., Wu, W., Kurtek, S., Klassen, E. &
  Marron, J.S. (2011). "Registration of Functional Data
  Using the Fisher-Rao Metric." arXiv:1103.3817.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
DTW finds the minimum-cost monotone alignment between two
series under a Sakoe-Chiba band (``|i - j| <= band * n``),
which bounds warping and prevents the pathological
"everything aligns to everything" degeneracy of
unconstrained warping. Cost is local squared distance; the
optimal path is recovered by backtracking the DP table.
The registration utility extracts the warp-induced phase
shift and amplitude scaling (least-squares on the aligned
pairs) — the minimal honest decomposition of
amplitude-vs-phase variation without claiming a full
Fisher-Rao SRVF pipeline. Guards: band must keep a feasible
path (band*n >= |len diff| + 1), unequal lengths allowed,
zero-amplitude target fails closed. ``synth_dtw`` plants a
phase-shifted + amplitude-scaled copy plus an unrelated
control; the bench gates on the DTW cost being lower for
the warped twin, on recovered shift within tolerance, and
on the control showing a larger normalized distance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 20) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def dtw_cost(
    x: FloatArray,
    y: FloatArray,
    band: float = 0.25,
) -> dict[str, float | FloatArray]:
    """Sakoe-Chiba banded DTW distance + warp path."""
    vx = _as_series(x)
    vy = _as_series(y)
    # z-normalize: DTW on shape, not level/scale
    vx = (vx - np.mean(vx)) / np.std(vx)
    vy = (vy - np.mean(vy)) / np.std(vy)
    n, m = vx.size, vy.size
    b = max(int(np.ceil(band * max(n, m))), abs(n - m) + 1, 2)
    if b >= max(n, m):
        raise ValueError("band too wide")
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - b)
        hi = min(m, i + b)
        for j in range(lo, hi + 1):
            c = (vx[i - 1] - vy[j - 1]) ** 2
            dp[i, j] = c + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    if not np.isfinite(dp[n, m]):
        raise ValueError("no feasible warp path")
    # backtrack
    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        cand = np.array([dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]])
        k = int(np.argmin(cand))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i = i - 1
        else:
            j = j - 1
    path.reverse()
    pth = np.asarray(path, dtype=np.float64)
    out: dict[str, float | FloatArray] = {
        "cost": float(dp[n, m]),
        "cost_per_step": float(dp[n, m] / max(len(path), 1)),
        "path": pth,
    }
    return out


def warp_register(
    x: FloatArray,
    y: FloatArray,
    band: float = 0.25,
) -> dict[str, float]:
    """Phase shift + amplitude scale from the DTW alignment."""
    r = dtw_cost(x, y, band=band)
    path = np.asarray(r["path"])
    xi, yi = path[:, 0].astype(np.int64), path[:, 1].astype(np.int64)
    vx = _as_series(x)
    vy = _as_series(y)
    xa, ya = vx[xi], vy[yi]
    # shift: mean index delta; amplitude: LS scale y ~ s*x
    shift = float(np.mean(yi - xi))
    scale = float(np.sum(xa * ya) / np.sum(xa * xa)) if np.sum(xa * xa) > 1e-12 else 0.0
    resid = float(np.sqrt(np.mean((ya - scale * xa) ** 2)))
    out: dict[str, float] = {
        "shift": shift,
        "scale": scale,
        "aligned_rmse": resid,
        "cost_per_step": float(r["cost_per_step"]),
    }
    return out


def synth_dtw(
    seed: int = 20261231 + 355,
    n: int = 120,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC phase-shifted + scaled twin + unrelated control."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = np.sin(2 * np.pi * t / 40.0) + 0.3 * np.sin(2 * np.pi * t / 13.0)
    shift = 7
    warped = (
        1.6 * np.sin(2 * np.pi * (t - shift) / 40.0)
        + 0.5 * np.sin(2 * np.pi * (t - shift) / 13.0)
        + 0.02 * rng.standard_normal(n)
    )
    indep = 0.8 * np.sin(2 * np.pi * t / 23.0) + 0.02 * rng.standard_normal(n)
    return (
        base.astype(np.float64),
        warped.astype(np.float64),
        indep.astype(np.float64),
    )


def bench_dtw(seed: int = 20261231 + 355) -> dict[str, float]:
    base, warped, indep = synth_dtw(seed=seed)
    r_w = dtw_cost(base, warped)
    r_i = dtw_cost(base, indep)
    reg = warp_register(base, warped)
    ok = (
        r_w["cost_per_step"] < r_i["cost_per_step"]
        and abs(reg["shift"] - 7.0) < 4.0
        and 0.5 < reg["scale"] < 2.4
        and reg["aligned_rmse"] < 0.9
    )
    out: dict[str, float] = {
        "synthetic_dtw_cost_warped": float(r_w["cost_per_step"]),
        "synthetic_dtw_cost_indep": float(r_i["cost_per_step"]),
        "synthetic_dtw_shift_hat": reg["shift"],
        "synthetic_dtw_scale_hat": reg["scale"],
        "score": 1.0 if ok else 0.0,
    }
    return out
