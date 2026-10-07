"""Radix-partitioned hash join (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 737


def radix_join(left: np.ndarray, right: np.ndarray, bits: int = 2) -> list[tuple[int, int]]:
    """Partition both sides by low `bits` of hash, join within partitions."""
    nparts = 1 << bits
    out: list[tuple[int, int]] = []
    for p in range(nparts):
        lp = [x for x in left if hash(int(x)) % nparts == p]
        rp = {int(x) for x in right if hash(int(x)) % nparts == p}
        for x in lp:
            if int(x) in rp:
                out.append((int(x), int(x)))
    return out


def bench_radix_join(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    left = rng.randint(0, 100, 80)
    right = rng.randint(0, 100, 80)
    got = {t[0] for t in radix_join(left, right)}
    expect = set(int(x) for x in left) & set(int(x) for x in right)
    return {"synthetic_radix_exact": float(got == expect)}
