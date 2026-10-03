"""Simplicial homology H_k = ker d_k / im d_{k+1} via boundary ranks (SYNTHETIC)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def boundary_matrix(
    faces: Sequence[tuple[int, ...]], cells: Sequence[tuple[int, ...]]
) -> np.ndarray:
    """Boundary d: C_{dim faces} -> C_{dim cells} with entries +-1 over Z (real)."""
    m = np.zeros((len(cells), len(faces)))
    idx = {c: j for j, c in enumerate(cells)}
    for j, f in enumerate(faces):
        for i in range(len(f)):
            face = f[:i] + f[i + 1 :]
            m[idx[face], j] = (-1) ** i
    return m


def _gf2_rank(m: np.ndarray) -> int:
    a = (m != 0).astype(int) % 2
    rows = a.copy()
    rank = 0
    for col in range(rows.shape[1]):
        piv = np.nonzero(rows[:, col])[0]
        piv = piv[piv >= rank]
        if len(piv) == 0:
            continue
        rows[[rank, piv[0]]] = rows[[piv[0], rank]]
        rows[rank + 1 :] = (rows[rank + 1 :] + rows[rank] * rows[rank + 1 :, col : col + 1]) % 2
        rank += 1
    return rank


def betti(
    vertices: list[int], edges: list[tuple[int, int]], faces: list[tuple[int, int, int]]
) -> tuple[int, int, int]:
    """(b0, b1, b2) of a 2-d complex over Z2."""
    v = [(i,) for i in vertices]
    d1 = boundary_matrix(edges, v) if edges else np.zeros((len(v), 1))
    d2 = boundary_matrix(faces, edges) if faces else np.zeros((len(edges), 1))
    r1 = _gf2_rank(d1)
    r2 = _gf2_rank(d2) if faces else 0
    n_v, n_e, n_f = len(v), len(edges), len(faces)
    b0 = n_v - r1
    b1 = (n_e - r1) - r2
    b2 = n_f - r2
    return b0, b1, b2


def _bench_simplicial_homology(seed: int = 0) -> float:
    checks = []
    # tetrahedron boundary: b = (1, 0, 1)
    v = [0, 1, 2, 3]
    e = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    f = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    checks.append(betti(v, e, f) == (1, 0, 1))
    # hollow triangle (no face): b = (1, 1, 0)
    checks.append(betti([0, 1, 2], [(0, 1), (1, 2), (0, 2)], []) == (1, 1, 0))
    # filled triangle: b = (1, 0, 0)
    checks.append(betti([0, 1, 2], [(0, 1), (1, 2), (0, 2)], [(0, 1, 2)]) == (1, 0, 0))
    # two disjoint edges: b0 = 2
    checks.append(betti([0, 1, 2, 3], [(0, 1), (2, 3)], [])[:1] == (2,))
    # d1 @ d2 = 0 mod 2 on tetrahedron
    d1 = boundary_matrix(e, [(i,) for i in v])
    d2 = boundary_matrix(f, e)
    checks.append(int(np.max(np.abs((d1 @ d2) % 2))) == 0)
    # filled square (disc): contractible b = (1,0,0); chi = 4-5+2 = 1
    checks.append(
        betti([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)], [(0, 1, 2), (0, 2, 3)])
        == (1, 0, 0)
    )
    return float(sum(checks) / len(checks))


def bench_simplicial_homology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simplicial_homology": _bench_simplicial_homology(seed)}
