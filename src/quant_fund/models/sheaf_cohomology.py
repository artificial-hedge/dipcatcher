"""Cech cohomology of a sheaf on an open cover (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def cech_h1_dim(cover: list[set[int]], triples: list[tuple[int, int, int]]) -> int:
    """H^1 of the constant sheaf R on a cover of the circle/simplicial circle.

    cover: list of open sets as vertex sets; a 1-cocycle assigns reals to
    ordered pairwise overlaps; cocycle condition c_ij + c_jk = c_ik on
    triple overlaps. H^1 = ker(d1) / im(d0) where d0: C^0 -> C^1,
    (d0 a)_ij = a_j - a_i. Compute dim via ranks of the coboundary
    matrices built over the nerve.
    """
    n = len(cover)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if cover[i] & cover[j]]
    np_ = len(pairs)
    pidx = {p: k for k, p in enumerate(pairs)}
    # d0: np x n
    d0 = np.zeros((np_, n))
    for k, (i, j) in enumerate(pairs):
        d0[k, i] = -1.0
        d0[k, j] = 1.0
    # d1: ntriples x np over triple overlaps (i,j,k) all present in nerve
    ntrip = len(triples)
    d1 = np.zeros((ntrip, np_))
    for t, (i, j, k) in enumerate(triples):
        # (d1 c)_ijk = c_jk - c_ik + c_ij
        for s, p_ in ((1, (j, k)), (-1, (i, k)), (1, (i, j))):
            if p_ in pidx:
                d1[t, pidx[p_]] = s
    dim_z1 = np_ - np.linalg.matrix_rank(d1)
    dim_b1 = int(np.linalg.matrix_rank(d0))
    return int(dim_z1 - dim_b1)


def graph_h1(edges: list[tuple[int, int]], n_vertices: int) -> int:
    """First Betti number of a graph: cycle rank E - V + components."""
    parent = list(range(n_vertices))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    comps = len({find(v) for v in range(n_vertices)})
    return len(edges) - n_vertices + comps


def _bench_sheaf_cohomology(seed: int = 0) -> float:
    checks = []
    # circle covered by 3 arcs (each misses a different third of S^1):
    # nerve of the cover = triangle (all pairs and the triple overlap is
    # empty on a real cover, but the standard computation has H^1 = R).
    # Mimic the nerve: U0,U1,U2 pairwise overlapping, no triple overlap.
    u0, u1, u2 = {0, 1}, {1, 2}, {0, 2}
    # only pairwise overlaps exist -> triples list empty -> H^1 = Z^1 = all C^1
    # minus coboundaries: 3 pairs, d1 empty -> dim ker d1 = 3 - 0 = 3;
    # im d0 = rank 2 -> H^1 = 1 = dim H^1(S^1, R)
    checks.append(cech_h1_dim([u0, u1, u2], []) == 1)
    # cover of an interval (contractible) by 2 opens with one overlap:
    # nerve = edge; H^1 = 0
    v0, v1 = {0, 1}, {1, 2}
    checks.append(cech_h1_dim([v0, v1], []) == 0)
    # figure-eight as a 1-complex: two triangles sharing vertex 0 -> H1 = 2
    fig8 = [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4)]
    checks.append(graph_h1(fig8, 5) == 2)
    # theta graph (two vertices joined by 3 edges): H1 = 2
    theta = [(0, 1), (0, 1), (0, 1)]
    checks.append(graph_h1(theta, 2) == 2)
    # filled triangle (contractible): 3 opens with a common triple overlap
    # -> d1 kills the cocycle -> H^1 = 0
    w0, w1, w2 = {0, 1, 3}, {1, 2, 3}, {0, 2, 3}
    checks.append(cech_h1_dim([w0, w1, w2], [(0, 1, 2)]) == 0)
    return float(sum(checks) / len(checks))


def bench_sheaf_cohomology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheaf_cohomology": _bench_sheaf_cohomology(seed)}
