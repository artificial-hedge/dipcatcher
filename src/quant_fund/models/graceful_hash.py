"""Grace hash join: spill partitions to 'disk' (lists) when build > budget (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 738


def graceful_join(
    left: np.ndarray, right: np.ndarray, budget: int, _depth: int = 0
) -> list[tuple[int, int]]:
    """Partition by hash until each fits budget; recurse on overflow."""
    out: list[tuple[int, int]] = []
    if len(left) <= budget or _depth > 8:
        probe = set(int(x) for x in right)
        for x in left:
            if int(x) in probe:
                out.append((int(x), int(x)))
        return out
    nparts = 2
    lparts: list[list[int]] = [[] for _ in range(nparts)]
    rparts: list[list[int]] = [[] for _ in range(nparts)]
    for x in left:
        lparts[hash(int(x)) % nparts].append(int(x))
    for x in right:
        rparts[hash(int(x)) % nparts].append(int(x))
    for p in range(nparts):
        out.extend(graceful_join(np.array(lparts[p]), np.array(rparts[p]), budget, _depth + 1))
    return out


def bench_graceful_hash(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    left = rng.randint(0, 200, 150)
    right = rng.randint(0, 200, 150)
    got = {t[0] for t in graceful_join(left, right, budget=10)}
    expect = set(int(x) for x in left) & set(int(x) for x in right)
    return {"synthetic_grace_exact": float(got == expect)}
