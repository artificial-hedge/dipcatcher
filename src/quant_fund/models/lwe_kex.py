"""LWE key exchange (Regev/NewHope-style, toy parameters)."""

import numpy as np

_SEED = 20261231 + 590


def _lwe_kex(rng: np.random.RandomState, q: int = 3329, n: int = 16) -> bool:
    A = rng.randint(0, q, (n, n))
    sa = rng.randint(-1, 2, n)
    ea = rng.randint(-1, 2, n)
    sb = rng.randint(-1, 2, n)
    eb = rng.randint(-1, 2, n)
    ua = (A @ sa + ea) % q
    ub = (A.T @ sb + eb) % q
    ka = (ub @ sa) % q
    kb = (ua @ sb) % q
    # reconcile: high bit should agree up to error |e·s| bounded by 2n < q/4
    ra = int(ka >= q // 4 and ka < 3 * q // 4)
    rb = int(kb >= q // 4 and kb < 3 * q // 4)
    return ra == rb


def bench_lwe_kex(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = sum(_lwe_kex(np.random.RandomState(rng.randint(2**31))) for _ in range(200))
    return {"synthetic_kex_agreement": ok / 200}
