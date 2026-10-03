"""Singular homology on a finite grid complex: H0 = components, H1 = loops (SYNTHETIC)."""

from __future__ import annotations

from collections.abc import Sequence


def graph_betti(n_vertices: int, edges: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """(b0, b1) for a 1-d simplicial complex: b1 = E - V + b0."""
    parent = list(range(n_vertices))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for u, v in edges:
        parent[find(u)] = find(v)
    b0 = len({find(i) for i in range(n_vertices)})
    b1 = len(edges) - n_vertices + b0
    return b0, b1


def _bench_singular_homology(seed: int = 0) -> float:
    checks = []
    # S^1 model: 4-cycle -> (1, 1)
    checks.append(graph_betti(4, [(0, 1), (1, 2), (2, 3), (3, 0)]) == (1, 1))
    # figure-eight: two loops -> (1, 2)
    checks.append(graph_betti(5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]) == (1, 2))
    # two disjoint circles -> (2, 2)
    checks.append(graph_betti(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]) == (2, 2))
    # tree -> (1, 0): contractible
    checks.append(graph_betti(4, [(0, 1), (0, 2), (0, 3)]) == (1, 0))
    # cylinder-ish ladder: 2 squares -> b1 = 8-6+1 = 3? compute: V=6 E=7 -> b1=2
    checks.append(
        graph_betti(6, [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]) == (1, 2)
    )
    # theta graph: two paths in parallel -> (1, 2)
    checks.append(graph_betti(4, [(0, 1), (1, 3), (0, 2), (2, 3), (0, 3)]) == (1, 2))
    return float(sum(checks) / len(checks))


def bench_singular_homology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_singular_homology": _bench_singular_homology(seed)}
