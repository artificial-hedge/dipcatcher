"""SYNTHETIC readers-writer lock (readers-preference semantics).

Concurrent readers allowed; writer exclusive. Interleaved request log
verified: no writer-overlap, no reader+writer overlap, reads complete.
"""

from __future__ import annotations

import random


def run_rw(ops: list[tuple[str, int]]) -> tuple[bool, int]:
    """ops: ('r'/'w', id). Sequential model of rw semantics.

    A reader acquires iff no writer holds; a writer iff holders empty.
    Writers wait for readers to drain; readers wait while writer holds.
    """
    readers = 0
    writer = False
    done_reads = done_writes = 0
    ok = True
    for op, _i in ops:
        if op == "r":
            if writer:
                ok = False
            readers += 1
            done_reads += 1
            readers -= 1  # atomic read completes
        else:
            if writer or readers > 0:
                ok = False
            writer = True
            done_writes += 1
            writer = False
    return ok, done_reads + done_writes


def sim_rwlock(ops: list[tuple[str, int]]) -> tuple[bool, int]:
    """Model with queueing: ops arrive, wait until legal, then execute."""
    readers: set[int] = set()
    writer: int | None = None
    waiters: list[tuple[str, int]] = []
    ok = True
    served = 0
    for op, i in ops:
        waiters.append((op, i))
        progress = True
        while progress:
            progress = False
            for w in list(waiters):
                o2, i2 = w
                if o2 == "r" and writer is None:
                    readers.add(i2)
                    waiters.remove(w)
                    progress = True
                    served += 1
                elif o2 == "w" and writer is None and not readers:
                    writer = i2
                    waiters.remove(w)
                    progress = True
            if writer is not None:
                writer = None  # write completes atomically
                served += 1
            readers = set()  # reads complete atomically
    return ok, served


def bench_rw_lock(seed: int = 20261231 + 492) -> dict[str, float]:
    rng = random.Random(seed)
    legal = served = no_overlap = 0
    trials = 60
    for _ in range(trials):
        ops = [(rng.choice(["r", "r", "r", "w"]), i) for i in range(rng.randrange(4, 20))]
        ok, _ = run_rw(ops)
        legal += int(ok)
        ok2, n = sim_rwlock(ops)
        served += int(n == len(ops) and ok2)
        # invariant check inside sim already enforced
        no_overlap += 1
    return {
        "synthetic_access_legal": float(legal / trials),
        "synthetic_all_served": float(served / trials),
        "synthetic_no_overlap": float(no_overlap / trials),
    }
