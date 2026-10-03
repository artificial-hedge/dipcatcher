"""AddressSanitizer-style shadow memory: poisoned redzone detection."""

import numpy as np

_SEED = 20261231 + 672


def bench_asan_shadow(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    detected = 0.0
    trials = 50
    for _ in range(trials):
        n = rng.randint(8, 20)
        shadow = np.zeros(n + 4)  # 2-byte redzones each side
        shadow[:2] = shadow[-2:] = 1  # poisoned
        # simulate random access; count caught OOB
        idx = rng.randint(-2, n + 2)
        mapped = idx + 2
        oob = idx < 0 or idx >= n
        caught = bool(shadow[mapped]) if 0 <= mapped < len(shadow) else False
        detected += float(oob == caught)
    return {"synthetic_asan_detects": detected / trials}
