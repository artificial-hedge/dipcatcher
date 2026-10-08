"""BGP path-vector simulator: announcement propagation, loop rejection, (SYNTHETIC)
local-pref selection, convergence to loop-free routes."""

import numpy as np

_SEED = 20261231 + 610


def bgp_converge(
    edges: list[tuple[int, int]], origin: int, n_as: int
) -> dict[int, tuple[int, ...]]:
    """Return best AS_PATH per AS reachable from origin."""
    adj: dict[int, list[int]] = {i: [] for i in range(n_as)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    best: dict[int, tuple[int, ...]] = {origin: (origin,)}
    changed = True
    while changed:
        changed = False
        for a in list(best):
            for b in adj[a]:
                if b in best[a]:  # loop check: own AS already in path
                    continue
                cand = best[a] + (b,)
                if (
                    b not in best
                    or len(cand) < len(best[b])
                    or (len(cand) == len(best[b]) and cand < best[b])
                ):
                    best[b] = cand
                    changed = True
    return best


def bench_bgp_pathvec(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(30):
        n = rng.randint(6, 12)
        edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.rand() < 0.35]
        if not edges:
            continue
        origin = 0
        routes = bgp_converge(edges, origin, n)
        # oracle: BFS shortest path
        import collections

        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        dist = {origin: 0}
        q = collections.deque([origin])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        good = all(len(routes[v]) - 1 == dist[v] for v in dist)
        good = good and all(len(set(routes[v])) == len(routes[v]) for v in routes)
        ok += good
    return {"synthetic_bgp_optimal": ok / 30}
