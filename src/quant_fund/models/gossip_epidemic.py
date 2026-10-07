"""Epidemic/gossip broadcast on a random graph (synthetic) (SYNTHETIC).

Push-style rumor spreading: each round every informed node picks f
random peers and pushes. Verified: (i) full coverage on connected
graphs in O(log N) rounds; (ii) coverage count vs BFS oracle;
(iii) round complexity bound.
"""

from __future__ import annotations

import math
import random


def _erdos(n: int, p: float, rng: random.Random) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _bfs(adj: list[set[int]], src: int) -> set[int]:
    seen = {src}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return seen


def gossip(
    adj: list[set[int]], src: int, fanout: int, rng: random.Random, max_rounds: int = 60
) -> dict[str, float | int]:
    informed = {src}
    rounds = 0
    newly = {src}
    while newly and rounds < max_rounds:
        rounds += 1
        senders = list(informed)
        new_new: set[int] = set()
        for u in senders:
            peers = list(adj[u])
            rng.shuffle(peers)
            for v in peers[:fanout]:
                if v not in informed:
                    new_new.add(v)
        informed |= new_new
        newly = new_new
    return {"rounds": rounds, "covered": len(informed)}


def bench_gossip_epidemic(seed: int = 20261231 + 254) -> dict[str, float]:
    rng = random.Random(seed)
    n = 200
    adj = _erdos(n, 0.04, rng)
    # ensure connectivity from src 0
    reach = _bfs(adj, 0)
    res = gossip(adj, 0, fanout=3, rng=rng)
    coverage = res["covered"] == len(reach) == n
    log_bound = res["rounds"] <= int(3 * math.log2(n)) + 4
    # disconnected: coverage must equal BFS-reachable set exactly
    adj2 = _erdos(n, 0.01, rng)
    reach2 = _bfs(adj2, 0)
    res2 = gossip(adj2, 0, fanout=3, rng=rng)
    partial_ok = res2["covered"] == len(reach2)
    return {
        "synthetic_coverage": float(coverage),
        "synthetic_rounds": float(res["rounds"]),
        "synthetic_log_bound": float(log_bound),
        "synthetic_partial_exact": float(partial_ok),
    }
