"""SIS-based compression hash: h(x) = A x mod q, x in {0,1}^m (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 593


def sis_hash(A: np.ndarray, x: np.ndarray, q: int) -> np.ndarray:
    return np.asarray((A @ x) % q)


def bench_sis_hash(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    q, n, m = 331, 8, 24
    ok = 0
    trials = 60
    for _ in range(trials):
        A = rng.randint(0, q, (n, m))
        x1 = rng.randint(0, 2, m)
        x2 = rng.randint(0, 2, m)
        h1, h2 = sis_hash(A, x1, q), sis_hash(A, x2, q)
        # deterministic + compression; collision only if h1==h2 with x1!=x2
        same = np.array_equal(h1, h2) and not np.array_equal(x1, x2)
        if not same and np.array_equal(sis_hash(A, x1, q), h1):
            ok += 1
    return {"synthetic_sis_binding": ok / trials}
