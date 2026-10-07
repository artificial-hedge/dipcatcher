"""Watchdog supervisor: kicks tasks that miss their check-in window (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 654


def bench_watchdog_task(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    caught = 0
    trials = 40
    for _ in range(trials):
        last_kick = {i: 0 for i in range(4)}
        deadline = 10
        frozen = rng.randint(4)  # one task freezes
        detect = -1
        for t in range(1, 60):
            for i in range(4):
                if i != frozen and rng.rand() < 0.9:
                    last_kick[i] = t
            for i in range(4):
                if t - last_kick[i] > deadline and detect < 0:
                    detect = t
        if detect > 0 and last_kick[frozen] <= detect - deadline:
            caught += 1
    return {"synthetic_watchdog_detects": caught / trials}
