"""Shared-memory bank-conflict counter for stride-p patterns (SYNTHETIC)."""

import math

import numpy as np

_SEED = 20261231 + 688


def conflicts(addrs: np.ndarray, n_banks: int = 32) -> int:
    """Max number of distinct addresses mapping to the same bank (conflict degree)."""
    banks = [int(a) % n_banks for a in addrs]
    uniq: dict[int, set[int]] = {}
    for a, b in zip(addrs, banks, strict=True):
        uniq.setdefault(b, set()).add(int(a))
    return max((len(v) for v in uniq.values()), default=0)


def bench_bank_conflict(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        stride = int(rng.randint(1, 9))
        addrs = (np.arange(32) * stride).astype(int)
        deg = conflicts(addrs)
        # exact degree: the 32 lanes hit 32/gcd(stride,32) distinct
        # banks, so the busiest bank holds gcd(stride,32) addresses
        ok += float(deg == math.gcd(stride, 32))
    return {"synthetic_bank_degree": ok / trials}
