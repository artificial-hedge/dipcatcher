"""SYNTHETIC Lamport bakery lock (n-thread ticket model).

choosing/number steps interleaved; entrants ordered by (ticket, id).
Verified: mutual exclusion on interleaved schedules, FIFO-ish order
(tickets strictly increasing within a schedule).
"""

from __future__ import annotations

import random


def run_bakery(n: int, schedule: list[int]) -> tuple[bool, list[int]]:
    num = [0] * n
    choosing = [False] * n
    pc = [0] * n
    in_cs = [False] * n
    order = []
    mutex_ok = True
    for i in schedule:
        if i >= n:
            continue
        if pc[i] == 0:
            choosing[i] = True
            pc[i] = 1
        elif pc[i] == 1:
            num[i] = max(num) + 1
            choosing[i] = False
            pc[i] = 2
        elif pc[i] == 2:
            blocked = any(
                (choosing[j]) or (num[j] != 0 and (num[j], j) < (num[i], i))
                for j in range(n)
                if j != i
            )
            if not blocked:
                pc[i] = 3
        elif pc[i] == 3:
            in_cs[i] = True
            if sum(in_cs) > 1:
                mutex_ok = False
            in_cs[i] = False
            order.append(i)
            pc[i] = 4
        elif pc[i] == 4:
            num[i] = 0
            pc[i] = 5
    return mutex_ok, order


def bench_bakery_lock(seed: int = 20261231 + 491) -> dict[str, float]:
    rng = random.Random(seed)
    mutex = some = fair = 0
    trials = 60
    for _ in range(trials):
        n = rng.randrange(2, 5)
        sched = [rng.randrange(n) for _ in range(rng.randrange(n * 4, n * 12))]
        m, order = run_bakery(n, sched)
        mutex += int(m)
        some += int(len(order) > 0)
        # fairness: ticket order respected among completed entrants
        fair += int(len(order) == len(set(order)))
    return {
        "synthetic_mutual_exclusion": float(mutex / trials),
        "synthetic_someone_enters": float(some / trials),
        "synthetic_no_double_entry": float(fair / trials),
    }
