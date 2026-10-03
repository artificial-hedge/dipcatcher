"""Hungarian algorithm on small square costs vs exhaustive oracle."""

import itertools

import numpy as np

_SEED = 20261231 + 622


def hungarian(cost: np.ndarray) -> tuple[float, list[int]]:
    n = len(cost)
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    assign = [0] * n
    for j in range(1, n + 1):
        assign[p[j] - 1] = j - 1
    return -float(v[0]), assign


def bench_kuhn_munkres(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(25):
        n = rng.randint(3, 7)
        C = rng.uniform(0, 10, (n, n))
        val, assign = hungarian(C)
        best = min(
            sum(C[i, perm[i]] for i in range(n)) for perm in itertools.permutations(range(n))
        )
        ok += abs(val - best) < 1e-6 and sum(C[i, a] for i, a in enumerate(assign)) - best < 1e-6
    return {"synthetic_hungarian_optimal": float(ok) / 25}
