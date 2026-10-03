"""Consistent hashing with virtual nodes (synthetic).

Karger-style ring hashing: each physical node gets V virtual tokens
on a 2^32 ring; keys map to the successor token's owner. Verified:
(i) minimal remapping — removing one node moves ~1/N of keys;
(ii) load balance — max/mean ratio bounded with V tokens; (iii)
deterministic stable mapping.
"""

from __future__ import annotations

import bisect
import hashlib
import random


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode(), usedforsecurity=False).hexdigest(), 16) & 0xFFFFFFFF


class Ring:
    def __init__(self, nodes: list[str], vnodes: int = 100) -> None:
        self.ring: list[tuple[int, str]] = []
        for n in nodes:
            for i in range(vnodes):
                self.ring.append((_h(f"{n}#{i}"), n))
        self.ring.sort()
        self.keys = [k for k, _ in self.ring]

    def owner(self, key: str) -> str:
        i = bisect.bisect_left(self.keys, _h(key))
        return self.ring[i % len(self.ring)][1]


def bench_consistent_hash(seed: int = 20261231 + 253) -> dict[str, float]:
    rng = random.Random(seed)
    keys = [f"k{i}" for i in range(3000)]
    nodes = [f"n{j}" for j in range(8)]
    r = Ring(nodes, vnodes=150)
    base_map = {k: r.owner(k) for k in keys}
    # remove one node
    nodes2 = nodes[:-1]
    r2 = Ring(nodes2, vnodes=150)
    moved = sum(1 for k in keys if r2.owner(k) != base_map[k])
    moved_frac = moved / len(keys)
    # expected ~1/8 = 0.125; bound generously
    minimal = moved_frac < 0.25
    # balance: max bucket share vs mean
    counts = {n: 0 for n in nodes}
    for k in keys:
        counts[r.owner(k)] += 1
    mx = max(counts.values())
    mean = len(keys) / len(nodes)
    balance = mx / mean
    balanced = balance < 2.5
    # determinism
    stable = all(r.owner(k) == base_map[k] for k in rng.sample(keys, 100))
    return {
        "synthetic_moved_frac": float(moved_frac),
        "synthetic_minimal_remap": float(minimal),
        "synthetic_max_over_mean": float(balance),
        "synthetic_balanced": float(balanced),
        "synthetic_stable": float(stable),
    }
