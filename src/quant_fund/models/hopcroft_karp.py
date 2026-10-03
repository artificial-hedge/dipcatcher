"""Hopcroft-Karp bipartite maximum matching (BFS layering + DFS augment)."""

import collections

import numpy as np

_SEED = 20261231 + 621


def hopcroft_karp(adj: list[list[int]], n_r: int) -> dict[int, int]:
    n_l = len(adj)
    pair_l = [-1] * n_l
    pair_r = [-1] * n_r
    dist = [0] * n_l

    def bfs() -> bool:
        q: collections.deque = collections.deque()
        for u in range(n_l):
            if pair_l[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                w = pair_r[v]
                if w == -1:
                    found = True
                elif dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            w = pair_r[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                pair_l[u] = v
                pair_r[v] = u
                return True
        dist[u] = -1
        return False

    while bfs():
        for u in range(n_l):
            if pair_l[u] == -1:
                dfs(u)
    return {u: pair_l[u] for u in range(n_l) if pair_l[u] != -1}


def _aug(adj: list[list[int]], pair_r: list[int], u: int, seen: set[int]) -> bool:
    for v in adj[u]:
        if v in seen:
            continue
        seen.add(v)
        if pair_r[v] == -1 or _aug(adj, pair_r, pair_r[v], seen):
            pair_r[v] = u
            return True
    return False


def bench_hopcroft_karp(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(30):
        n_l, n_r = rng.randint(3, 8, 2)
        adj = [[int(x) for x in np.flatnonzero(rng.rand(n_r) < 0.4)] for _ in range(n_l)]
        got = len(hopcroft_karp(adj, n_r))
        pair_r = [-1] * n_r
        want = sum(_aug(adj, pair_r, u, set()) for u in range(n_l))
        ok += got == want
    return {"synthetic_hk_maximal": ok / 30}
