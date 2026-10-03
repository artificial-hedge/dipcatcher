"""Neighbor-joining tree reconstruction (wave 284).

Greedy NJ on a distance matrix recovers the correct unrooted topology for
additive quartet distances — verified against planted ((a,b),(c,d)) trees.
"""

import numpy as np

_SEED = 20261231 + 788


def _dist_from_tree(tree: dict[int, list[tuple[int, float]]], taxa: list[int]) -> np.ndarray:
    import collections

    n = len(taxa)
    d = np.zeros((n, n))
    for i, s in enumerate(taxa):
        # bfs from s
        dist = {s: 0.0}
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for v, w in tree[u]:
                if v not in dist:
                    dist[v] = dist[u] + w
                    q.append(v)
        for j, t in enumerate(taxa):
            d[i, j] = dist[t]
    return d


def nj(d: np.ndarray, names: list[str]) -> set[frozenset[str]]:
    # returns set of cherry pairs (frozensets of two leaf names)
    labels = names[:]
    dm = {i: {j: d[i, j] for j in range(len(names))} for i in range(len(names))}
    active = list(range(len(names)))
    cherries: set[frozenset[str]] = set()
    while len(active) > 3:
        n = len(active)
        r = {i: sum(dm[i][j] for j in active) for i in active}
        best: tuple[int, int] | None = None
        best_q = float("inf")
        for x in range(n):
            for y in range(x + 1, n):
                i, j = active[x], active[y]
                q_val = (n - 2) * dm[i][j] - r[i] - r[j]
                if q_val < best_q:
                    best_q, best = q_val, (i, j)
        if best is None:
            break
        i, j = best
        cherries.add(frozenset([labels[i], labels[j]]))
        u = max(active) + 1
        dm[u] = {}
        for k in active:
            dm[u][k] = dm[k][u] = 0.5 * (dm[i][k] + dm[j][k] - dm[i][j])
        active = [a for a in active if a not in best] + [u]
        labels.append(f"u{u}")
    # final triple joins at one center: any pair of original leaves there is a cherry
    leaf_labels = set(names)
    finals = [labels[a] for a in active]
    orig = [x for x in finals if x in leaf_labels]
    if len(orig) == 2:
        cherries.add(frozenset(orig))
    return cherries


def bench_nj_tree(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(4):
        # tree: cherry (a,b) and cherry (c,d) joined by internal edge
        il = rng.uniform(0.05, 0.3)
        tree: dict[int, list[tuple[int, float]]] = {i: [] for i in range(6)}
        # leaves 0..3, internals 4,5
        for leaf, w in [(0, 0.1), (1, 0.2), (2, 0.15), (3, 0.25)]:
            parent = 4 if leaf < 2 else 5
            tree[leaf].append((parent, w))
            tree[parent].append((leaf, w))
        tree[4].append((5, il))
        tree[5].append((4, il))
        d = _dist_from_tree(tree, [0, 1, 2, 3])
        got = nj(d, ["a", "b", "c", "d"])
        want = {frozenset(["a", "b"]), frozenset(["c", "d"])}
        ok += int(got == want)
    return {"synthetic_nj_topo": float(ok == 4)}
