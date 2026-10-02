"""Fréchet canon: discrete Fréchet distance between polygonal
curves — the O(nm) dynamic-program leash coupling, plus the
coupling path extraction. Bench: known values on simple curves,
symmetry, triangle-inequality sanity vs Euclidean endpoints.
All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _d(a: FloatArray, b: FloatArray) -> float:
    return float(np.linalg.norm(a - b))


def frechet_dist(P: FloatArray, Q: FloatArray) -> tuple[float, FloatArray]:
    """Discrete Fréchet distance + the coupling matrix."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)
    ca[0, 0] = _d(P[0], Q[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], _d(P[i], Q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], _d(P[0], Q[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                _d(P[i], Q[j]),
            )
    return float(ca[n - 1, m - 1]), np.asarray(ca, dtype=np.float64)


def frechet_path(P: FloatArray, Q: FloatArray) -> list[tuple[int, int]]:
    """Backtrack a min-leash coupling through the DP matrix."""
    d, ca = frechet_dist(P, Q)
    i, j = len(P) - 1, len(Q) - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        c = []
        if i > 0:
            c.append((ca[i - 1, j], i - 1, j))
        if j > 0:
            c.append((ca[i, j - 1], i, j - 1))
        if i > 0 and j > 0:
            c.append((ca[i - 1, j - 1], i - 1, j - 1))
        _, i, j = min(c)
        path.append((i, j))
    path.reverse()
    return path


def bench_frechet(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    # identical curves → 0
    t = np.linspace(0, 2 * np.pi, 50)
    C = np.stack([np.cos(t), np.sin(t)], axis=1)
    d0, _ = frechet_dist(C, C)
    out["synthetic_frechet_self"] = d0
    # shifted curve → distance = shift magnitude
    C2 = C + np.array([0.5, 0.0])
    d1, _ = frechet_dist(C, C2)
    out["synthetic_frechet_shift_err"] = abs(d1 - 0.5)
    # symmetry
    d2, _ = frechet_dist(C2, C)
    out["synthetic_frechet_sym_err"] = abs(d1 - d2)
    # collinear different sampling: same line different speeds
    L1 = np.stack([np.linspace(0, 1, 11), np.zeros(11)], axis=1)
    L2 = np.stack([np.linspace(0, 1, 101), np.zeros(101)], axis=1)
    d3, _ = frechet_dist(L1, L2)
    out["synthetic_frechet_resample"] = d3
    # endpoint bound: distance ≥ max endpoint gap
    A = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    B = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    d4, _ = frechet_dist(A, B)
    out["synthetic_frechet_detour"] = d4
    # coupling path validity: monotone, ends correct
    path = frechet_path(A, B)
    out["synthetic_frechet_path_len"] = float(len(path))
    out["synthetic_frechet_path_ends"] = float(path[0] == (0, 0) and path[-1] == (2, 2))
    return out
