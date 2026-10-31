"""NTRU encrypt/decrypt in Z_q[x]/(x^N - 1), toy parameters (SYNTHETIC)."""

import contextlib

import numpy as np

_SEED = 20261231 + 591


def _mul(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    c = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            c[(i + j) % n] += a[i] * b[j]
    return c


def _circulant(f: np.ndarray, n: int) -> np.ndarray:
    M = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            M[i, (i - j) % n] = f[j]
    return M


def _inv_modq(M: np.ndarray, q: int, n: int) -> np.ndarray | None:
    """Gauss-Jordan inversion of M over Z_q (q prime)."""
    aug = np.concatenate([M % q, np.eye(n, dtype=np.int64)], axis=1) % q
    for col in range(n):
        piv = -1
        for r in range(col, n):
            if aug[r, col] % q != 0:
                piv = r
                break
        if piv < 0:
            return None
        aug[[col, piv]] = aug[[piv, col]]
        inv = pow(int(aug[col, col]), q - 2, q)
        aug[col] = (aug[col] * inv) % q
        for r in range(n):
            if r != col and aug[r, col] != 0:
                aug[r] = (aug[r] - aug[r, col] * aug[col]) % q
    return aug[:, n:] % q


def _ntru_roundtrip(rng: np.random.RandomState, n: int = 11, p: int = 3, q: int = 199) -> bool:
    # keygen: f = 1 + p*F invertible mod p trivially (=1), and mod q via
    # standard construction fp = 1, fq computed by extended Euclidean on
    # circulant — toy: pick f s.t. fq exists (check by solving circulant sys).
    f = np.ones(n, dtype=np.int64)
    fq: np.ndarray | None = None
    for _ in range(50):
        F = rng.randint(-1, 2, n)
        f = p * F
        f[0] += 1
        Minv = _inv_modq(_circulant(f % q, n), q, n)
        if Minv is not None:
            fq = Minv[:, 0] % q
            break
    if fq is None:
        return False
    g = rng.randint(-1, 2, n)
    h = (p * _mul(fq, g, n)) % q
    m = rng.randint(0, p, n)
    r = rng.randint(-1, 2, n)
    c = (_mul(r, h, n) + m) % q
    a = _mul(f, c, n) % q
    a = np.where(a > q // 2, a - q, a)
    dec = a % p
    return np.array_equal(dec % p, m % p)


def bench_ntru_toy(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    trials = 40
    for _ in range(trials):
        with contextlib.suppress(Exception):
            ok += _ntru_roundtrip(np.random.RandomState(rng.randint(2**31)))
    return {"synthetic_ntru_roundtrip": ok / trials}
