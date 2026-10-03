"""Many-to-one hospital/resident deferred acceptance with quotas."""

import numpy as np

_SEED = 20261231 + 624


def gale_chu(students: np.ndarray, hosp: np.ndarray, quota: np.ndarray) -> dict[int, list[int]]:
    """students[i] prefs over hospitals; hosp[h] prefs over students."""
    n_s = len(students)
    nxt = [0] * n_s
    assigned: dict[int, list[int]] = {h: [] for h in range(len(hosp))}
    free = list(range(n_s))
    while free:
        s = free.pop()
        if nxt[s] >= len(students[s]):
            continue
        h = int(students[s, nxt[s]])
        nxt[s] += 1
        rank = {int(v): i for i, v in enumerate(hosp[h])}
        if s not in rank:
            continue
        cur = assigned[h]
        if len(cur) < quota[h]:
            cur.append(s)
        else:
            worst = max(cur, key=lambda t: rank[t])
            if rank[s] < rank[worst]:
                cur.remove(worst)
                cur.append(s)
                free.append(worst)
            else:
                free.append(s)
    return assigned


def bench_gale_chu(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(40):
        n_s, n_h = rng.randint(5, 10), rng.randint(2, 5)
        quota = rng.randint(1, 3, n_h)
        students = np.stack([rng.permutation(n_h) for _ in range(n_s)])
        hosp = np.stack([rng.permutation(n_s) for _ in range(n_h)])
        out = gale_chu(students, hosp, quota)
        valid = all(len(v) <= quota[h] for h, v in out.items())
        flat = [s for v in out.values() for s in v]
        ok += valid and len(flat) == len(set(flat))
    return {"synthetic_gc_valid": ok / 40}
