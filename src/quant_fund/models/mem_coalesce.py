"""Memory coalescing: count 128-byte transactions for a warp's accesses."""

import numpy as np

_SEED = 20261231 + 689


def transactions(addrs: np.ndarray, line: int = 128) -> int:
    segs = {int(a) // line for a in addrs}
    return len(segs)


def bench_mem_coalesce(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        # contiguous: 32 threads * 4B = 128B -> 1 transaction
        base = int(rng.randint(0, 4096))
        contiguous = base + 4 * np.arange(32)
        scattered = rng.randint(0, 4096, 32)
        t_c = transactions(contiguous)
        t_s = transactions(scattered)
        ok += float(t_c <= 2 and t_s >= t_c)
    return {"synthetic_coalesce_better": ok / trials}
