"""Gale-Shapley deferred acceptance + blocking-pair oracle (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 620


def gale_shapley(men: np.ndarray, women: np.ndarray) -> dict[int, int]:
    """men[i] = ranked preference list over women; women[j] = ranked list over men."""
    n = len(men)
    nxt = [0] * n
    engaged: dict[int, int] = {}  # woman -> man
    free = list(range(n))
    while free:
        m = free.pop()
        w = int(men[m, nxt[m]])
        nxt[m] += 1
        if w not in engaged:
            engaged[w] = m
        else:
            cur = engaged[w]
            if int(np.where(women[w] == m)[0][0]) < int(np.where(women[w] == cur)[0][0]):
                engaged[w] = m
                free.append(cur)
            else:
                free.append(m)
    return {m: w for w, m in engaged.items()}


def _is_stable(match: dict[int, int], men: np.ndarray, women: np.ndarray) -> bool:
    inv = {w: m for m, w in match.items()}
    for m, w in match.items():
        for w2 in men[m, : int(np.where(men[m] == w)[0][0])]:
            m2 = inv[int(w2)]
            if int(np.where(women[int(w2)] == m)[0][0]) < int(np.where(women[int(w2)] == m2)[0][0]):
                return False  # (m, w2) blocking pair
    return True


def bench_gale_shapley(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(40):
        n = rng.randint(3, 8)
        men = np.stack([rng.permutation(n) for _ in range(n)])
        women = np.stack([rng.permutation(n) for _ in range(n)])
        ok += _is_stable(gale_shapley(men, women), men, women)
    return {"synthetic_gs_stable": ok / 40}
