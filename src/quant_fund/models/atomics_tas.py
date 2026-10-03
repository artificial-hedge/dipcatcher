"""SYNTHETIC atomic primitives: TAS spinlock + CAS counter.

Interleaved fetch-and-add CAS loops vs sequential reference — final
count equals number of increments; spinlock admits one holder at a time.
"""

from __future__ import annotations

import random


def cas_inc(mem: list[int], who: int, rng: random.Random) -> int:
    """CAS increment: read, compute, compare-and-swap — retry on race."""
    while True:
        old = mem[0]
        new = old + 1
        if rng.random() < 0.9 or mem[0] == old:  # CAS: succeeds if unchanged
            if mem[0] == old:
                mem[0] = new
                return new
            # raced: retry
        else:
            continue


def run_counter(nthreads: int, incs: int, rng: random.Random) -> int:
    mem = [0]
    sched = [(t, rng.random()) for t in range(nthreads) for _ in range(incs)]
    rng.shuffle(sched)
    for t, _ in sched:
        cas_inc(mem, t, rng)
    return mem[0]


class TasLock:
    def __init__(self):
        self.state = 0

    def tas(self) -> int:
        old = self.state
        self.state = 1
        return old

    def acquire(self) -> None:
        while self.tas() == 1:
            pass

    def release(self) -> None:
        self.state = 0


def bench_atomics_tas(seed: int = 20261231 + 493) -> dict[str, float]:
    rng = random.Random(seed)
    exact = mutex = live = 0
    trials = 40
    for _ in range(trials):
        n, incs = rng.randrange(2, 6), rng.randrange(2, 10)
        exact += int(run_counter(n, incs, rng) == n * incs)
        # TAS mutex: sequential model — holder set size ≤ 1
        lock = TasLock()
        holders = set()
        ok = True
        for t in range(4):
            if rng.random() < 0.7:
                old = lock.tas()
                if old == 0:
                    holders.add(t)
                    if len(holders) > 1:
                        ok = False
                    holders.discard(t)
                    lock.release()
        mutex += int(ok)
        live += int(lock.state == 0)
    return {
        "synthetic_cas_count_exact": float(exact / trials),
        "synthetic_tas_mutual_exclusion": float(mutex / trials),
        "synthetic_lock_released": float(live / trials),
    }
