"""Tutte-Berge formula: max matching = (n + min_S(|S| - odd(G-S)))/2 (SYNTHETIC)."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations


def _components(adj: list[set[int]], verts: Iterable[int]) -> list[set[int]]:
    verts_set = set(verts)
    seen: set[int] = set()
    comps = []
    for s in verts_set:
        if s in seen:
            continue
        comp = set()
        stack = [s]
        while stack:
            u = stack.pop()
            if u in comp or u not in verts_set:
                continue
            comp.add(u)
            stack.extend(adj[u])
        seen |= comp
        comps.append(comp)
    return comps


def max_matching_brute(adj: list[set[int]]) -> int:
    n = len(adj)
    best = 0

    def rec(covered: set[int], size: int) -> None:
        nonlocal best
        if size + (n - len(covered)) // 2 <= best:
            return
        v = next((i for i in range(n) if i not in covered), None)
        if v is None:
            best = max(best, size)
            return
        for u in list(adj[v]):
            if u not in covered:
                rec(covered | {v, u}, size + 1)
        rec(covered | {v}, size)

    rec(set(), 0)
    return best


def tutte_berge_value(adj: list[set[int]]) -> int:
    n = len(adj)
    worst = n
    verts = list(range(n))
    for r in range(n + 1):
        for s_set in combinations(verts, r):
            s = set(s_set)
            rest = [v for v in verts if v not in s]
            odd = sum(1 for c in _components(adj, rest) if len(c) % 2 == 1)
            worst = min(worst, len(s) - odd)
    return (n + worst) // 2


def _mk(edges: list[tuple[int, int]], n: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _bench_tutte_berge(seed: int = 0) -> float:
    checks = []
    petersen = _mk(
        [(i, (i + 1) % 5) for i in range(5)]
        + [(i + 5, (i + 2) % 5 + 5) for i in range(5)]
        + [(i, i + 5) for i in range(5)],
        10,
    )
    checks.append(max_matching_brute(petersen) == 5)
    checks.append(tutte_berge_value(petersen) == 5)
    path = _mk([(i, i + 1) for i in range(5)], 6)
    checks.append(tutte_berge_value(path) == 3 == max_matching_brute(path))
    star = _mk([(0, i) for i in range(1, 6)], 6)
    checks.append(tutte_berge_value(star) == 1 == max_matching_brute(star))
    # odd cycle C5: max matching 2
    c5 = _mk([(i, (i + 1) % 5) for i in range(5)], 5)
    checks.append(tutte_berge_value(c5) == 2 == max_matching_brute(c5))
    k4 = _mk([(a, b) for a in range(4) for b in range(a + 1, 4)], 4)
    checks.append(tutte_berge_value(k4) == 2)
    return float(sum(checks) / len(checks))


def bench_tutte_berge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tutte_berge": _bench_tutte_berge(seed)}
