"""Rate-monotonic schedulability via the Liu-Layland utilization bound (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 651


def ll_bound(tasks: list[tuple[float, float]]) -> bool:
    """tasks: (exec, period). U <= n(2^(1/n)-1) sufficient."""
    n = len(tasks)
    u = sum(e / p for e, p in tasks)
    return u <= n * (2 ** (1 / n) - 1)


def bench_rms_scheduler(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    sound = 0.0
    trials = 60
    for _ in range(trials):
        n = rng.randint(2, 6)
        tasks = [(float(rng.randint(1, 4)), float(rng.randint(4, 20))) for _ in range(n)]
        pred = ll_bound(tasks)
        prio = sorted(range(n), key=lambda i: tasks[i][1])
        feasible = True
        for i in range(n):
            r_t = tasks[i][0]
            for _ in range(20):
                new_r = tasks[i][0] + sum(
                    np.ceil(r_t / tasks[j][1]) * tasks[j][0]
                    for j in prio
                    if j != i and tasks[j][1] <= tasks[i][1]
                )
                if new_r == r_t:
                    break
                r_t = new_r
            if r_t > tasks[i][1]:
                feasible = False
        # Liu-Layland sufficient: pred => RTA feasible
        sound += 1.0 if (not pred or feasible) else 0.0
    return {"synthetic_rms_ll_sound": float(sound) / trials}
