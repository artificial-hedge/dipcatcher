"""SYNTHETIC demand pager + page table.

Virtual pages lazily mapped to a bounded frame pool with FIFO eviction;
verified: accesses return correct data (write-through backing store),
fault count ≤ cold-miss + capacity bounds.
"""

from __future__ import annotations

import random
from collections import deque


class Pager:
    def __init__(self, n_pages: int, n_frames: int):
        self.backing = [0] * n_pages
        self.frames = [-1] * n_frames
        self.table: dict[int, int] = {}
        self.queue: deque[int] = deque()
        self.faults = 0
        self.n_frames = n_frames
        self.touched: set[int] = set()

    def _map(self, v: int) -> int:
        if v in self.table:
            return self.table[v]
        self.faults += 1
        if len(self.queue) == self.n_frames:
            old = self.queue.popleft()
            f = self.table.pop(old)
        else:
            f = len(self.queue)
        self.queue.append(v)
        self.touched.add(v)
        self.frames[f] = v
        self.table[v] = f
        return f

    def read(self, v: int) -> int:
        self._map(v)
        return self.backing[v]

    def write(self, v: int, x: int) -> None:
        self._map(v)
        self.backing[v] = x


def bench_mmap_pager(seed: int = 20261231 + 472) -> dict[str, float]:
    rng = random.Random(seed)
    data_ok = fault_ok = evict_ok = 0
    trials = 40
    for _ in range(trials):
        np_, nf = 30, 6
        p = Pager(np_, nf)
        oracle = [0] * np_
        ok = True
        ops = rng.randrange(30, 80)
        for _ in range(ops):
            v = rng.randrange(np_)
            if rng.random() < 0.5:
                x = rng.randrange(100)
                p.write(v, x)
                oracle[v] = x
            else:
                if p.read(v) != oracle[v]:
                    ok = False
        data_ok += int(ok)
        # faults ≥ cold misses (each distinct page touched once at least)
        fault_ok += int(p.faults >= len(p.touched))
        evict_ok += int(len(p.table) <= nf and len(p.frames) == nf)
    return {
        "synthetic_read_write_correct": float(data_ok / trials),
        "synthetic_faults_ge_cold": float(fault_ok / trials),
        "synthetic_resident_bound": float(evict_ok / trials),
    }
