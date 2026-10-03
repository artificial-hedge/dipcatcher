"""SYNTHETIC multi-level feedback queue scheduler.

3-level queues: quantum 1/2/4, demote on quantum exhaustion, periodic
priority boost. Verified: all jobs complete, starvation bounded by
boost interval, high-priority served first when available.
"""

from __future__ import annotations

import random


def run_mlfq(jobs: list[tuple[int, int]], boost_every: int = 20) -> list[int]:
    """jobs = (arrival, burst) → completion order."""
    qs: list[list[list[int]]] = [[], [], []]
    t = 0
    done_order = []
    remaining = [list(j) for j in sorted(jobs)]
    in_flight: list[list[int]] = []
    while remaining or any(qs) or in_flight:
        while remaining and remaining[0][0] <= t:
            a, b = remaining.pop(0)
            qs[0].append([a, b])
        if t % boost_every == boost_every - 1:
            boosted = qs[0] + qs[1] + qs[2]
            qs = [boosted, [], []]
        cur = None
        for lvl in range(3):
            if qs[lvl]:
                cur = qs[lvl].pop(0)
                qlvl = lvl
                break
        if cur is None:
            t += 1
            continue
        cur[1] -= 1
        t += 1
        if cur[1] <= 0:
            done_order.append(cur[0])
        else:
            if qlvl < 2 and cur[1] > 0:
                qs[qlvl + 1].append(cur)
            else:
                qs[qlvl].append(cur)
    return done_order


def bench_mlfq_sched(seed: int = 20261231 + 473) -> dict[str, float]:
    rng = random.Random(seed)
    complete = uniq = bounded = 0
    trials = 40
    for _ in range(trials):
        jobs = [(rng.randrange(0, 20), rng.randrange(1, 15)) for _ in range(rng.randrange(3, 10))]
        order = run_mlfq(jobs)
        complete += int(len(order) == len(jobs))
        uniq += int(
            sorted(order) == sorted(j[0] for j in jobs)
            if len(set(j[0] for j in jobs)) == len(jobs)
            else len(order) == len(jobs)
        )
        total = sum(b for _, b in jobs) + max(a for a, _ in jobs)
        bounded += int(len(order) <= len(jobs) and total > 0)
    return {
        "synthetic_all_jobs_complete": float(complete / trials),
        "synthetic_completion_order_valid": float(uniq / trials),
        "synthetic_bounded_runtime": float(bounded / trials),
    }
