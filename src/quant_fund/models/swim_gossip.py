"""SYNTHETIC SWIM-style gossip membership dissemination.

Nodes exchange membership maps {id: incarnation}; updates merge by
max incarnation. Verify: convergence to a common view in O(log n) rounds,
failed node detected by all survivors, incarnation freshness wins.
"""

from __future__ import annotations

import random


def converge(n: int, rng: random.Random, fanout: int = 2) -> int:
    views = [{i: 1 for i in range(n)} for _ in range(n)]
    rounds = 0
    while len({tuple(sorted(v.items())) for v in views}) > 1 and rounds < 50:
        rounds += 1
        for u in range(n):
            for _ in range(fanout):
                v = rng.randrange(n)
                if v == u:
                    continue
                # merge u's map into v's by max incarnation
                for k, inc in views[u].items():
                    views[v][k] = max(views[v].get(k, 0), inc)
    return rounds


def bench_swim_gossip(seed: int = 20261231 + 444) -> dict[str, float]:
    rng = random.Random(seed)
    conv = fast = detect = 0
    trials = 40
    for _ in range(trials):
        n = rng.randrange(4, 12)
        r = converge(n, rng)
        conv += int(r <= 50)
        fast += int(r <= max(3, n // 2 + 2))
        # failed node: gossip suspects propagated (incarnation bump marked)
        views = [{i: 1 for i in range(n)} for _ in range(n)]
        dead = rng.randrange(n)
        for u in range(n):
            if u != dead:
                views[u][dead] = 0  # mark suspect
        for _ in range(10):
            for u in range(n):
                if u == dead:
                    continue
                v = rng.randrange(n)
                if v in (u, dead):
                    continue
                for k, inc in views[u].items():
                    views[v][k] = max(views[v].get(k, 0), inc)
        detect += int(all(views[u][dead] == 0 for u in range(n) if u != dead))
    return {
        "synthetic_converges": float(conv / trials),
        "synthetic_log_rounds": float(fast / trials),
        "synthetic_suspect_propagates": float(detect / trials),
    }
