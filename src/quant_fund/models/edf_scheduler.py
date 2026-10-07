"""EDF scheduling: preemptive earliest-deadline-first, deadline-miss metric (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 650


def edf_run(tasks: list[tuple[int, int, int]], horizon: int) -> int:
    """tasks: (release, exec_time, deadline). Returns deadline misses."""
    remaining = {i: e for i, (_, e, _) in enumerate(tasks)}
    done = set()
    misses = 0
    for t in range(horizon + 1):
        # mark misses
        for i, (_r, _, d) in enumerate(tasks):
            if i not in done and t > d:
                misses += 1
                done.add(i)
        ready = [i for i, (r, _, _) in enumerate(tasks) if r <= t and i not in done]
        if not ready:
            continue
        cur = min(ready, key=lambda i: tasks[i][2])
        remaining[cur] -= 1
        if remaining[cur] == 0:
            done.add(cur)
    return misses


def bench_edf_scheduler(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    hits = 0
    trials = 60
    for _ in range(trials):
        n = rng.randint(2, 5)
        tasks = []
        for _ in range(n):
            r = rng.randint(0, 5)
            e = rng.randint(1, 4)
            d = r + e + rng.randint(3, 10)
            tasks.append((int(r), int(e), int(d)))
        hits += edf_run(tasks, 40) == 0
    return {"synthetic_edf_miss_free": hits / trials}
