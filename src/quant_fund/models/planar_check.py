"""Planarity obstruction checks: Euler bound + K5/K3,3 detection (SYNTHETIC)."""

from __future__ import annotations

import itertools

Graph = dict[int, frozenset[int]]


def edges_of(g: Graph) -> frozenset[frozenset[int]]:
    out = set()
    for v, ns in g.items():
        for w in ns:
            out.add(frozenset({v, w}))
    return frozenset(out)


def euler_bound_ok(g: Graph) -> bool:
    """Simple planar graph on v>=3 has e <= 3v - 6."""
    v = len(g)
    e = len(edges_of(g))
    if v < 3:
        return True
    return e <= 3 * v - 6


def has_k5_subgraph(g: Graph) -> bool:
    """Exact K5 vertex subset (not subdivision)."""
    vs = list(g)
    for c in itertools.combinations(vs, 5):
        sub = {v: g[v] & set(c) for v in c}
        if all(len(sub[v]) == 4 for v in sub):
            return True
    return False


def has_k33_subgraph(g: Graph) -> bool:
    vs = list(g)
    for a in itertools.combinations(vs, 3):
        rest = [v for v in vs if v not in a]
        for b in itertools.combinations(rest, 3):
            if all(all(w in g[u] for w in b) for u in a):
                return True
    return False


def _bench_planar_check(seed: int = 0) -> float:
    checks = []
    k5 = {i: frozenset(j for j in range(5) if j != i) for i in range(5)}
    checks.append(not euler_bound_ok(k5))  # e=10 > 9
    checks.append(has_k5_subgraph(k5))
    k33 = {i: frozenset({3, 4, 5}) for i in range(3)} | {
        i: frozenset({0, 1, 2}) for i in range(3, 6)
    }
    checks.append(euler_bound_ok(k33))  # e=9 <= 12: bound doesn't rule it out
    checks.append(has_k33_subgraph(k33))
    tri = {0: frozenset({1, 2}), 1: frozenset({0, 2}), 2: frozenset({0, 1})}
    checks.append(euler_bound_ok(tri))
    checks.append(not has_k5_subgraph(tri))
    return float(sum(checks) / len(checks))


def bench_planar_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_planar_check": _bench_planar_check(seed)}
