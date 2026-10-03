"""SYNTHETIC bidirectional Dijkstra shortest path.

Forward + backward alternating search with meet-point total; distances
verified against one-directional Dijkstra on random weighted graphs.
"""

from __future__ import annotations

import heapq
import random


def bidirectional_dijkstra(
    edges_f: dict[int, list[tuple[int, float]]],
    edges_b: dict[int, list[tuple[int, float]]],
    s: int,
    t: int,
) -> float:
    """Alternating bidirectional Dijkstra; returns shortest distance."""
    df: dict[int, float] = {s: 0.0}
    db: dict[int, float] = {t: 0.0}
    pf = [(0.0, s)]
    pb = [(0.0, t)]
    seen_f: set[int] = set()
    seen_b: set[int] = set()
    best = 10**18.0
    while pf or pb:
        if pf:
            d, u = heapq.heappop(pf)
            if u not in seen_f:
                seen_f.add(u)
                if u in db:
                    best = min(best, d + db[u])
                for v, w in edges_f.get(u, []):
                    if v not in seen_f and d + w < df.get(v, 10**18):
                        df[v] = d + w
                        heapq.heappush(pf, (df[v], v))
            if seen_b and pf and pf[0][0] + (pb[0][0] if pb else 0) >= best:
                pf.clear()
        if pb:
            d, u = heapq.heappop(pb)
            if u not in seen_b:
                seen_b.add(u)
                if u in df:
                    best = min(best, d + df[u])
                for v, w in edges_b.get(u, []):
                    if v not in seen_b and d + w < db.get(v, 10**18):
                        db[v] = d + w
                        heapq.heappush(pb, (db[v], v))
            if seen_f and pb and pb[0][0] + (pf[0][0] if pf else 0) >= best:
                pb.clear()
    return best


def _dijkstra(edges: dict[int, list[tuple[int, float]]], s: int, t: int) -> float:
    g = {s: 0.0}
    pq = [(0.0, s)]
    done = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        if u == t:
            return d
        for v, w in edges.get(u, []):
            if v not in done and d + w < g.get(v, 10**18):
                g[v] = d + w
                heapq.heappush(pq, (g[v], v))
    return g.get(t, 10**18)


def bench_bidirectional_dijkstra(seed: int = 20261231 + 523) -> dict[str, float]:
    rng = random.Random(seed)
    agree = 0
    n = 60
    for _ in range(n):
        nv = rng.randrange(6, 15)
        edges_f: dict[int, list[tuple[int, float]]] = {}
        edges_b: dict[int, list[tuple[int, float]]] = {}
        for u in range(nv):
            for v in range(nv):
                if u != v and rng.random() < 0.3:
                    w = rng.uniform(0.1, 5)
                    edges_f.setdefault(u, []).append((v, w))
                    edges_b.setdefault(v, []).append((u, w))
        s, t = 0, nv - 1
        bi = bidirectional_dijkstra(edges_f, edges_b, s, t)
        di = _dijkstra(edges_f, s, t)
        agree += int(abs(bi - di) < 1e-9 or (bi >= 10**17 and di >= 10**17))
    return {"synthetic_distance_exact": agree / n}
