"""Quorum read/write: R + W > N consistency (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 744


def quorum_write(val: int, n: int, w: int) -> dict[int, int]:
    """Write to first w replicas."""
    return {i: val for i in range(w)}


def quorum_read(replicas: dict[int, int], n: int, r: int, total_val: int) -> int:
    """Read r replicas; return the value (any read of written replica gets it)."""
    seen = [replicas.get(i, -1) for i in range(r)]
    return max(seen)


def bench_quorum_rw(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 50
    for _ in range(trials):
        n = int(rng.randint(3, 10))
        w = int(rng.randint(1, n))
        r = n - w + 1  # R + W = N + 1 → any read quorum intersects write quorum
        v = int(rng.randint(100))
        reps = quorum_write(v, n, w)
        got = quorum_read(reps, n, r, v)
        ok += float(got == v)
    return {"synthetic_quorum_reads": ok / trials}
