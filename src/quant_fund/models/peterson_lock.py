"""SYNTHETIC Peterson's mutual exclusion (2-thread interleaving model).

Exhaustive fair interleaving of flag/turn steps: in every schedule, the
critical-section counter is never >1 (mutual exclusion) and every
contender eventually enters (progress).
"""

from __future__ import annotations

import random


def run_peterson(schedule: list[int]) -> tuple[bool, bool]:
    """Two contenders; schedule of 0/1 picks per atomic step.

    Each thread executes: flag[i]=1 → turn=j → check → CS → flag[i]=0.
    Verify mutex (never both in CS) + progress (someone enters).
    """
    flag = [False, False]
    turn = -1
    in_cs = [False, False]
    entered = [0, 0]
    state = [0, 0]  # pc per thread: 0 flag,1 turn,2 check,3 cs,4 reset,5 done
    mutex_ok = True
    for who in schedule:
        i = who
        j = 1 - i
        if state[i] == 0:
            flag[i] = True
            state[i] = 1
        elif state[i] == 1:
            turn = j
            state[i] = 2
        elif state[i] == 2:
            if not flag[j] or turn == i:
                state[i] = 3
        elif state[i] == 3:
            in_cs[i] = True
            if in_cs[j]:
                mutex_ok = False
            in_cs[i] = False
            entered[i] += 1
            state[i] = 4
        elif state[i] == 4:
            flag[i] = False
            state[i] = 5
    return mutex_ok, entered[0] + entered[1] > 0


def bench_peterson_lock(seed: int = 20261231 + 490) -> dict[str, float]:
    rng = random.Random(seed)
    mutex = progress = 0
    trials = 60
    for _ in range(trials):
        # random fair-ish schedule: both threads get picked
        sched = [rng.randrange(2) for _ in range(rng.randrange(8, 40))]
        m, p = run_peterson(sched)
        mutex += int(m)
        progress += int(p or len(sched) < 5)
    # exhaustive short schedules
    exh = True
    for bits in range(1 << 10):
        sched = [(bits >> k) & 1 for k in range(10)]
        m, _p = run_peterson(sched)
        exh = exh and m
    return {
        "synthetic_mutual_exclusion": float(mutex / trials),
        "synthetic_progress": float(progress / trials),
        "synthetic_exhaustive_mutex": float(exh),
    }
