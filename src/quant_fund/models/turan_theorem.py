"""Turan's theorem: Turan graph T(n,r) is extremal K_{r+1}-free (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def turan_graph(n: int, r: int) -> list[set[int]]:
    """Complete r-partite balanced graph."""
    parts: list[list[int]] = [[] for _ in range(r)]
    for v in range(n):
        parts[v % r].append(v)
    adj: list[set[int]] = [set() for _ in range(n)]
    for p in range(r):
        for q in range(p + 1, r):
            for u in parts[p]:
                adj[u].update(parts[q])
            for v in parts[q]:
                adj[v].update(parts[p])
    return adj


def edge_count(adj: list[set[int]]) -> int:
    return sum(len(a) for a in adj) // 2


def has_k_clique(adj: list[set[int]], k: int) -> bool:
    n = len(adj)
    for combo in combinations(range(n), k):
        if all(b in adj[a] for a, b in combinations(combo, 2)):
            return True
    return False


def _bench_turan_theorem(seed: int = 0) -> float:
    checks = []
    # Mantel: ex(10, K3) = 25; T(10,2) = K_{5,5}
    t = turan_graph(10, 2)
    checks.append(edge_count(t) == 25 and not has_k_clique(t, 3))
    # ex(n, K_{r+1}) = (1 - 1/r) n^2 / 2 formula on n=9, r=3 -> 27
    t93 = turan_graph(9, 3)
    checks.append(edge_count(t93) == 27)
    checks.append(not has_k_clique(t93, 4))
    # adding any edge inside a part creates K_{r+1}
    t83 = turan_graph(8, 3)
    parts: list[list[int]] = [[], [], []]
    for v in range(8):
        parts[v % 3].append(v)
    bad = True
    for p in parts:
        for a, b in combinations(p, 2):
            g2 = [set(s) for s in t83]
            g2[a].add(b)
            g2[b].add(a)
            if not has_k_clique(g2, 4):
                bad = False
    checks.append(bad)
    # T(6,3) has n^2/3 edges = 12
    checks.append(edge_count(turan_graph(6, 3)) == 12)
    # complete r-partite covers all cross edges: T(8,2) = K_{4,4} has 16
    checks.append(edge_count(turan_graph(8, 2)) == 16)
    return float(sum(checks) / len(checks))


def bench_turan_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_turan_theorem": _bench_turan_theorem(seed)}
