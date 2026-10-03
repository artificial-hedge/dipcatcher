"""Cayley graphs of finite groups: diameter, connectivity (SYNTHETIC)."""

from __future__ import annotations


def cayley_edges(g_elems: frozenset, mul, gens: frozenset) -> dict:
    return {x: {mul(x, g) for g in gens} for x in g_elems}


def bfs_diameter(adj: dict) -> int:
    """Diameter over connected graph."""
    from collections import deque

    nodes = list(adj)
    worst = 0
    for s in nodes:
        dist = {s: 0}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj.get(u, set()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        worst = max(worst, max(dist.values(), default=0))
    return worst


def connected(adj: dict, start) -> bool:
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj.get(u, set()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(adj)


def _bench_cayley_graph(seed: int = 0) -> float:
    checks = []
    z4 = frozenset(range(4))

    def mul4(a, b):
        return (a + b) % 4

    adj = cayley_edges(z4, mul4, frozenset({1}))
    checks.append(connected(adj, 0))
    checks.append(bfs_diameter(adj) == 3)  # directed C4: longest dist 0->3 is 3
    # Z4 with gen {2}: disconnected (order-2 element only)
    adj2 = cayley_edges(z4, mul4, frozenset({2}))
    checks.append(not connected(adj2, 0))
    # S3 with r,f: connected diameter? diameter of Cayley graph S3 = 3
    s3 = frozenset({(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)})

    def mul3(p, q):
        return tuple(p[q[i]] for i in range(3))

    adj3 = cayley_edges(s3, mul3, frozenset({(1, 2, 0), (1, 0, 2)}))
    checks.append(connected(adj3, (0, 1, 2)))
    checks.append(bfs_diameter(adj3) == 2)  # every element of S3 within 2 generators
    return float(sum(checks) / len(checks))


def bench_cayley_graph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cayley_graph": _bench_cayley_graph(seed)}
