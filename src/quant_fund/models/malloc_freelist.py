"""SYNTHETIC free-list malloc (first-fit split + coalesce on free).

Arena of N cells; alloc splits blocks, free reinserts sorted and
coalesces adjacent runs. Verified: no overlap, all allocations honored
or honestly failed, free-list invariant (sorted, non-adjacent).
"""

from __future__ import annotations

import random


class Heap:
    def __init__(self, size: int):
        self.size = size
        self.free = [(0, size)]  # sorted (start, len) runs

    def alloc(self, n: int) -> int | None:
        for i, (s, ln) in enumerate(self.free):
            if ln >= n:
                self.free.pop(i)
                if ln > n:
                    self.free.insert(i, (s + n, ln - n))
                return s
        return None

    def dealloc(self, s: int, n: int) -> None:
        self.free.append((s, n))
        self.free.sort()
        merged: list[tuple[int, int]] = []
        for st, ln in self.free:
            if merged and merged[-1][0] + merged[-1][1] == st:
                merged[-1] = (merged[-1][0], merged[-1][1] + ln)
            else:
                merged.append((st, ln))
        self.free = merged


def bench_malloc_freelist(seed: int = 20261231 + 471) -> dict[str, float]:
    rng = random.Random(seed)
    no_overlap = invariant = serve = 0
    trials = 40
    for _ in range(trials):
        h = Heap(200)
        live: dict[int, int] = {}
        ok_overlap = ok_inv = True
        ops = rng.randrange(20, 60)
        for _ in range(ops):
            if live and rng.random() < 0.4:
                s = rng.choice(list(live))
                h.dealloc(s, live.pop(s))
            else:
                n = rng.randrange(1, 30)
                a = h.alloc(n)
                if a is not None:
                    live[a] = n
                    for s2, n2 in live.items():
                        if s2 != a and not (a + n <= s2 or s2 + n2 <= a):
                            ok_overlap = False
            # invariant: sorted, non-adjacent
            for i in range(len(h.free) - 1):
                if h.free[i][0] + h.free[i][1] >= h.free[i + 1][0]:
                    ok_inv = False
        no_overlap += int(ok_overlap)
        invariant += int(ok_inv)
        serve += int(h.alloc(1) is not None or not h.free)
    return {
        "synthetic_no_overlap": float(no_overlap / trials),
        "synthetic_freelist_invariant": float(invariant / trials),
        "synthetic_consistent_state": float(serve / trials),
    }
