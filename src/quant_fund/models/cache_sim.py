"""Set-associative LRU cache simulator — SYNTHETIC.

Verified: cold misses on unique lines, LRU eviction order on a
conflict trace, hit rate non-decreasing in capacity.
"""

from __future__ import annotations


class Cache:
    def __init__(self, sets: int, ways: int, line: int = 64) -> None:
        self.sets, self.ways, self.line = sets, ways, line
        self.tab: list[list[int]] = [[] for _ in range(sets)]  # MRU..LRU tags
        self.hits = 0
        self.misses = 0

    def access(self, addr: int) -> bool:
        tag = addr // self.line
        s = tag % self.sets
        t = tag // self.sets
        row = self.tab[s]
        if t in row:
            row.remove(t)
            row.insert(0, t)
            self.hits += 1
            return True
        row.insert(0, t)
        if len(row) > self.ways:
            row.pop()
        self.misses += 1
        return False


def bench_cache_sim(seed: int = 20261231 + 331) -> dict[str, float]:
    trials = 30
    cold_ok = lru_ok = mono_ok = 0
    for _ in range(trials):
        sets, ways = 8, 2
        c = Cache(sets, ways)
        # cold: each unique line misses once
        uniq = [i * 64 for i in range(16)]
        for a in uniq:
            c.access(a)
        cold_ok += int(c.misses == 16 and c.hits == 0)
        # LRU order: fill a set's ways then access eldest → evict
        c2 = Cache(sets, ways)
        base_set = 3
        addrs = [(base_set + i * sets) * 64 for i in range(ways + 1)]
        for a in addrs[:ways]:
            c2.access(a)
        victim_was_there = c2.access(addrs[0])  # hit → now MRU
        c2.access(addrs[ways])  # evicts tag of addrs[1] (LRU)
        still = c2.access(addrs[0])
        lru_ok += int(victim_was_there and still)
        # monotonicity: capacity up → hits up on a strided trace
        trace = [((i % 40) * 64) for i in range(200)]
        h = []
        for w in (1, 2, 4):
            cc = Cache(8, w)
            for a in trace:
                cc.access(a)
            h.append(cc.hits)
        mono_ok += int(h[0] <= h[1] <= h[2])
    return {
        "synthetic_cold_misses": float(cold_ok / trials),
        "synthetic_lru_order": float(lru_ok / trials),
        "synthetic_capacity_monotone": float(mono_ok / trials),
    }
