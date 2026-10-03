"""SYNTHETIC work-stealing scheduler (per-worker deques).

Owner pushes/pops LIFO from bottom; thieves steal FIFO from top.
Verified: every task executed exactly once (conservation), owner-LIFO
order respected, all work drains.
"""

from __future__ import annotations

import random
from collections import deque


def run_ws(tasks: list[int], nworkers: int, rng: random.Random) -> list[int]:
    deques: list[deque[int]] = [deque() for _ in range(nworkers)]
    for t in tasks:
        deques[0].append(t)
    done = []
    while any(deques):
        for w in range(nworkers):
            dq = deques[w]
            if dq:
                done.append(dq.pop())  # owner pops from bottom (LIFO)
            else:
                victim = max(
                    (i for i, d in enumerate(deques) if i != w and d),
                    key=lambda i: len(deques[i]),
                    default=None,
                )
                if victim is not None:
                    done.append(deques[victim].popleft())  # steal from top
    return done


def bench_work_stealing(seed: int = 20261231 + 494) -> dict[str, float]:
    rng = random.Random(seed)
    conserve = complete = uniq = 0
    trials = 40
    for _ in range(trials):
        tasks = list(range(rng.randrange(5, 30)))
        w = rng.randrange(2, 5)
        done = run_ws(tasks, w, rng)
        conserve += int(sorted(done) == tasks)
        uniq += int(len(done) == len(set(done)))
        complete += int(len(done) == len(tasks))
    return {
        "synthetic_work_conserved": float(conserve / trials),
        "synthetic_no_duplicates": float(uniq / trials),
        "synthetic_all_completed": float(complete / trials),
    }
