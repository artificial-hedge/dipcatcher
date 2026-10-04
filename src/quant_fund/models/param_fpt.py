"""Parameterized complexity: bounded-search-tree Vertex Cover FPT (SYNTHETIC bench)."""

from __future__ import annotations

import itertools


def vc_fpt(edges: list[tuple[int, int]], n: int, k: int) -> set[int] | None:
    """Bounded search tree: branch on an uncovered edge, depth <= k."""
    adj = edges
    cover: set[int] = set()

    def rec(rem_edges: list[tuple[int, int]], cov: set[int], depth: int) -> set[int] | None:
        if not rem_edges:
            return cov
        if depth == 0:
            return None
        u, v = rem_edges[0]
        for take in (u, v):
            rest = [e for e in rem_edges if take not in e]
            out = rec(rest, cov | {take}, depth - 1)
            if out is not None:
                return out
        return None

    return rec(adj, cover, k)


def vc_brute(edges: list[tuple[int, int]], n: int, k: int) -> bool:
    for comb in itertools.combinations(range(n), k):
        s = set(comb)
        if all(u in s or v in s for u, v in edges):
            return True
    return False


def kernel(
    edges: list[tuple[int, int]], n: int, k: int
) -> tuple[list[tuple[int, int]], set[int], int]:
    """Buss kernel: any vertex with degree > k must be in the cover."""
    cover: set[int] = set()
    rem = list(edges)
    changed = True
    k2 = k
    while changed:
        changed = False
        deg: dict[int, int] = {}
        for u, v in rem:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        for v, d in deg.items():
            if d > k2:
                cover.add(v)
                rem = [e for e in rem if v not in e]
                k2 -= 1
                changed = True
                break
    return rem, cover, k2


def _bench_param_fpt(seed: int = 0) -> float:
    checks = []
    path = [(0, 1), (1, 2), (2, 3)]
    sol = vc_fpt(path, 4, 2)
    checks.append(sol is not None and all(u in sol or v in sol for u, v in path))
    checks.append(vc_fpt(path, 4, 1) is None)
    checks.append(not vc_brute(path, 4, 1))
    tri = [(0, 1), (1, 2), (0, 2)]
    checks.append(vc_fpt(tri, 3, 2) is not None)
    checks.append(vc_fpt(tri, 3, 1) is None)
    star = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    rem, cov, k2 = kernel(star, 6, 3)
    checks.append(cov == {0} and rem == [] and k2 == 2)
    both = vc_fpt(path, 4, 2)
    checks.append(both is not None and len(both) <= 2)
    return sum(checks) / len(checks)


def bench_param_fpt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_param_fpt": _bench_param_fpt(seed)}
