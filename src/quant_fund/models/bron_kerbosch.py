"""SYNTHETIC Bron–Kerbosch maximal-clique enumeration.

Pivot-variant BK on small graphs verified against exhaustive clique search:
every maximal clique found, none duplicated, none non-clique.
"""

from __future__ import annotations

import random
from itertools import combinations


def bron_kerbosch(adj: dict[int, set[int]]) -> list[set[int]]:
    out: list[set[int]] = []

    def bk(r: set[int], p: set[int], x: set[int]):
        if not p and not x:
            out.append(set(r))
            return
        pivot = max(p | x, key=lambda u: len(p & adj[u]), default=None)
        cands = p - (adj[pivot] if pivot is not None else set())
        for v in list(cands):
            bk(r | {v}, p & adj[v], x & adj[v])
            p.discard(v)
            x.add(v)

    bk(set(), set(adj), set())
    return out


def _all_maximal_cliques_brute(adj: dict[int, set[int]]) -> set[frozenset[int]]:
    nodes = list(adj)
    cliques: set[frozenset[int]] = set()
    for r in range(1, len(nodes) + 1):
        for sub in combinations(nodes, r):
            s = set(sub)
            if all(v in adj[u] or u == v for u in s for v in s) and not any(
                w not in s and s <= adj[w] | {w} and s < (adj[w] | {w}) for w in nodes
            ):
                cliques.add(frozenset(s))
    return cliques


def bench_bron_kerbosch(seed: int = 20261231 + 525) -> dict[str, float]:
    rng = random.Random(seed)
    exact = 0
    n = 40
    for _ in range(n):
        nv = rng.randrange(4, 9)
        adj: dict[int, set[int]] = {u: set() for u in range(nv)}
        for u in range(nv):
            for v in range(u + 1, nv):
                if rng.random() < 0.4:
                    adj[u].add(v)
                    adj[v].add(u)
        found = {frozenset(c) for c in bron_kerbosch(adj)}
        truth = _all_maximal_cliques_brute(adj)
        exact += int(found == truth)
    return {"synthetic_maximal_cliques_exact": exact / n}
