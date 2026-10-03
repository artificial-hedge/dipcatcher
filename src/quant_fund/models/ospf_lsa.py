"""Link-state routing: LSA flood + SPF (Dijkstra) vs shortest-path oracle."""

import heapq

import numpy as np

_SEED = 20261231 + 710


def spf(adj: dict[int, list[tuple[int, int]]], src: int) -> dict[int, int]:
    dist = {src: 0}
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1 << 30):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, 1 << 30):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def bench_ospf_lsa(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        n = int(rng.randint(4, 10))
        adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.rand() < 0.4:
                    w = int(rng.randint(1, 10))
                    adj[i].append((j, w))
                    adj[j].append((i, w))
        got = spf(adj, 0)
        # oracle: Bellman-Ford
        exp = {i: 1 << 30 for i in range(n)}
        exp[0] = 0
        for _ in range(n):
            for u in adj:
                for v, w in adj[u]:
                    if exp[u] + w < exp[v]:
                        exp[v] = exp[u] + w
        ok += float(all(got.get(i, 1 << 30) == exp[i] for i in range(n)))
    return {"synthetic_spf_correct": ok / trials}
