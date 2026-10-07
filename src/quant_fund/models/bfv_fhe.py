"""Toy BFV homomorphic encryption over Z_q[x]/(x^N + 1) (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 592


def _mul_neg(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    c = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            k = i + j
            sign = 1 if k < n else -1
            c[k % n] += sign * a[i] * b[j]
    return c


def _bfv_check(rng: np.random.RandomState, n: int = 8, q: int = 4093, t: int = 17) -> bool:
    s = rng.randint(-1, 2, n)
    A = rng.randint(0, q, n)
    e = rng.randint(-1, 2, n)
    b = (-(_mul_neg(A, s, n)) + e) % q
    delta = q // t
    m1, m2 = rng.randint(0, t, n), rng.randint(0, t, n)
    e1, e2 = rng.randint(-1, 2, n), rng.randint(-1, 2, n)
    u = rng.randint(-1, 2, n)
    # encrypt m1 (RNS-free toy): ct = (b*u + e1 + delta*m1, A*u + e2)
    c0 = (_mul_neg(b, u, n) + e1 + delta * m1) % q
    c1 = (_mul_neg(A, u, n) + e2) % q
    # add m2 plaintext: c0 += delta*m2
    c0 = (c0 + delta * m2) % q
    dec = (c0 + _mul_neg(c1, s, n)) % q
    dec = np.where(dec > q // 2, dec - q, dec)
    rec = np.rint(dec / delta).astype(np.int64) % t
    return np.array_equal(rec, (m1 + m2) % t)


def bench_bfv_fhe(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = sum(_bfv_check(np.random.RandomState(rng.randint(2**31))) for _ in range(60))
    return {"synthetic_bfv_add_correct": ok / 60}
