"""Multi-index conjunctive lookup: intersect sorted posting lists."""

import numpy as np

_SEED = 20261231 + 739


def intersect_sorted(lists: list[list[int]]) -> list[int]:
    if not lists:
        return []
    lists = sorted(lists, key=len)
    out = lists[0]
    for lst in lists[1:]:
        s = set(lst)
        out = [x for x in out if x in s]
        if not out:
            break
    return out


def bench_index_intersect(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 300
    posting_a = sorted(rng.choice(n, 100, replace=False).tolist())
    posting_b = sorted(rng.choice(n, 100, replace=False).tolist())
    posting_c = sorted(rng.choice(n, 100, replace=False).tolist())
    got = intersect_sorted([posting_a, posting_b, posting_c])
    expect = sorted(set(posting_a) & set(posting_b) & set(posting_c))
    return {"synthetic_intersect_exact": float(got == expect)}
