"""Kahn topological layering + longest-path oracle on DAGs."""

import numpy as np

_SEED = 20261231 + 625


def kahn_layers(edges: list[tuple[int, int]], n: int) -> list[list[int]]:
    indeg = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    frontier = [i for i in range(n) if indeg[i] == 0]
    layers = []
    seen = 0
    while frontier:
        layers.append(sorted(frontier))
        seen += len(frontier)
        nxt = []
        for u in frontier:
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    nxt.append(v)
        frontier = nxt
    if seen < n:
        return []  # cycle
    return layers


def bench_topo_layers(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(40):
        n = rng.randint(4, 10)
        edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.rand() < 0.3]
        layers = kahn_layers(edges, n)
        pos = {v: i for i, layer in enumerate(layers) for v in layer}
        valid = all(pos[a] < pos[b] for a, b in edges)
        # longest-path layering oracle: layer = longest path len from source
        dist = {i: 0 for i in range(n)}
        for a, b in edges:
            dist[b] = max(dist[b], dist[a] + 1)
        layer_of = {v: dist[v] for v in dist}
        same = all(pos[v] == layer_of[v] for v in pos)
        ok += valid and same
    return {"synthetic_topo_correct": ok / 40}
