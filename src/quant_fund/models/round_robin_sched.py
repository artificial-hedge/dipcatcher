"""SYNTHETIC CPU scheduler comparison — FCFS vs SJF vs round robin.

Processes arrive with known bursts; TAT/response metrics verify classic
scheduling theory: SJF minimizes mean turnaround, RR minimizes response
for the time-sliced discipline, FCFS runs in arrival order.
"""

from __future__ import annotations

import random
from collections import deque


def _schedule(
    procs: list[tuple[int, int]], policy: str, q: int = 2
) -> dict[int, tuple[float, float]]:
    """procs: [(arrival, burst)]; returns pid -> (turnaround, response)."""
    n = len(procs)
    rem = [b for _a, b in procs]
    first_start = [-1.0] * n
    finish = [-1.0] * n
    t = 0.0
    done = 0
    ready: deque[int] = deque()
    i = 0  # next arrival index
    arrived: set[int] = set()

    def admit(upto: float) -> None:
        nonlocal i
        while i < n and procs[i][0] <= upto:
            ready.append(i)
            arrived.add(i)
            i += 1

    admit(t)
    idle_gaps = 0
    while done < n:
        if not ready:
            admit(t)
            t = max(t, procs[i][0]) if i < n else t
            idle_gaps += 1
            if idle_gaps > n * 10:
                break
            continue
        if policy == "fcfs":
            pid = ready.popleft()
            if first_start[pid] < 0:
                first_start[pid] = t
            t += rem[pid]
            rem[pid] = 0
            finish[pid] = t
            done += 1
            admit(t)
        elif policy == "sjf":
            pid = min(ready, key=lambda p: rem[p])
            ready.remove(pid)
            if first_start[pid] < 0:
                first_start[pid] = t
            t += rem[pid]
            rem[pid] = 0
            finish[pid] = t
            done += 1
            admit(t)
        else:  # rr
            pid = ready.popleft()
            if first_start[pid] < 0:
                first_start[pid] = t
            run = min(q, rem[pid])
            t += run
            rem[pid] -= run
            admit(t)
            if rem[pid] > 0:
                ready.append(pid)
            else:
                finish[pid] = t
                done += 1
    return {p: (finish[p] - procs[p][0], first_start[p] - procs[p][0]) for p in range(n)}


def bench_round_robin_sched(seed: int = 20261231 + 370) -> dict[str, float]:
    rng = random.Random(seed)
    sjf_best = rr_resp = fifo_ord = 0
    trials = 40
    for _ in range(trials):
        n = rng.randrange(4, 9)
        procs = sorted((rng.randrange(0, 6), rng.randrange(1, 10)) for _ in range(n))
        r_fcfs = _schedule(procs, "fcfs")
        r_sjf = _schedule(procs, "sjf")
        r_rr = _schedule(procs, "rr")
        tat = lambda r: sum(v[0] for v in r.values())  # noqa: E731
        sjf_best += int(tat(r_sjf) <= tat(r_fcfs) + 1e-9)
        # RR mean response ≤ FCFS mean response (bursts > quantum exist)
        resp = lambda r: sum(v[1] for v in r.values())  # noqa: E731
        rr_resp += int(resp(r_rr) <= resp(r_fcfs) + 1e-9)
        # FCFS first-start times are nondecreasing in arrival index
        starts = [r_fcfs[p][1] + procs[p][0] for p in range(n)]
        fifo_ord += int(all(starts[i] <= starts[i + 1] + 1e-9 for i in range(n - 1)))
    return {
        "synthetic_sjf_min_tat": float(sjf_best / trials),
        "synthetic_rr_min_response": float(rr_resp / trials),
        "synthetic_fcfs_order": float(fifo_ord / trials),
    }
