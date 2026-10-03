"""Loop unrolling: expand fixed-trip loop into blocks + residual."""

import numpy as np

_SEED = 20261231 + 726


def unrolled_sum(arr: np.ndarray, unroll: int) -> float:
    n = len(arr)
    acc = np.zeros(unroll)
    i = 0
    while i + unroll <= n:
        acc += arr[i : i + unroll]
        i += unroll
    total = float(acc.sum())
    while i < n:
        total += float(arr[i])
        i += 1
    return total


def bench_loop_unroll(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 50
    for _ in range(trials):
        arr = rng.normal(size=int(rng.randint(1, 100)))
        u = int(rng.randint(2, 9))
        ok += float(abs(unrolled_sum(arr, u) - arr.sum()) < 1e-8 * max(1.0, abs(arr.sum())))
    return {"synthetic_unroll_exact": ok / trials}
