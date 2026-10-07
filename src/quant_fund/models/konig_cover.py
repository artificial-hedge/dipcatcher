"""König's theorem: min vertex cover from bipartite max matching (SYNTHETIC)."""

import numpy as np

from quant_fund.models.hopcroft_karp import hopcroft_karp

_SEED = 20261231 + 623


def konig_cover(adj: list[list[int]], n_r: int) -> tuple[set[int], set[int]]:
    n_l = len(adj)
    match = hopcroft_karp(adj, n_r)
    inv = {v: u for u, v in match.items()}
    matched_l = set(match)
    # Z = vertices reachable from unmatched L by alternating paths
    z_l: set[int] = set()
    z_r: set[int] = set()
    stack = [u for u in range(n_l) if u not in matched_l]
    z_l.update(stack)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in inv and inv[v] != u and v not in z_r:  # non-matching edge
                z_r.add(v)
                w = inv[v]
                if w not in z_l:
                    z_l.add(w)
                    stack.append(w)
    cover_l = set(range(n_l)) - z_l
    cover_r = z_r
    return cover_l, cover_r


def bench_konig_cover(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(30):
        n_l, n_r = int(rng.randint(3, 7)), int(rng.randint(3, 7))
        adj = [[int(x) for x in np.flatnonzero(rng.rand(n_r) < 0.45)] for _ in range(n_l)]
        cover_l, cover_r = konig_cover(adj, n_r)
        valid = all(u in cover_l or v in cover_r for u in range(n_l) for v in adj[u])
        ok += valid and len(cover_l) + len(cover_r) == len(hopcroft_karp(adj, n_r))
    return {"synthetic_konig_min_cover": ok / 30}
