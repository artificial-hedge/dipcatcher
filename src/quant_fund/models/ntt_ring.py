"""NTT ring arithmetic — negacyclic polynomial multiplication in Z_q[x]/(x^n+1).

A real radix-2 Cooley-Tukey NTT over a 2n-th root of unity psi:
evaluation at odd powers psi^(2i+1) gives the negacyclic transform;
pointwise multiply + inverse transform recovers the (x^n+1) product.
q = 12289 admits a 2^12-th root, so n = 256 works with psi = g^24.
"""

import numpy as np

_SEED = 20261231 + 866

Q = 12289
N = 256
_G = 11  # primitive root of Z_12289


def _pow(a: int, e: int) -> int:
    return pow(a, e, Q)


PSI = _pow(_G, (Q - 1) // (2 * N))  # 2n-th root of unity
OMEGA = _pow(PSI, 2)  # n-th root


def _bitrev(a: np.ndarray) -> np.ndarray:
    n = a.size
    j = 0
    out = a.astype(np.int64).copy()
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            out[i], out[j] = out[j], out[i]
    return out


def ntt(a: np.ndarray, w: int = OMEGA) -> np.ndarray:
    """In-place radix-2 DFT over Z_q with root w (Cooley-Tukey)."""
    a = _bitrev(a)
    n = a.size
    m = 2
    while m <= n:
        wm = _pow(w, n // m)
        for k in range(0, n, m):
            wj = 1
            for j in range(m // 2):
                t = wj * a[k + j + m // 2] % Q
                u = a[k + j]
                a[k + j] = (u + t) % Q
                a[k + j + m // 2] = (u - t) % Q
                wj = wj * wm % Q
        m *= 2
    return a


def intt(a: np.ndarray, w: int = OMEGA) -> np.ndarray:
    """Inverse NTT: n^{-1} * DFT with w^{-1}."""
    n = a.size
    return (ntt(a, _pow(w, -1)) * _pow(n, -1)) % Q


def ntt_mul(a: np.ndarray, b: np.ndarray, q: int = Q) -> np.ndarray:
    """Negacyclic product a*b mod (x^N + 1, q) via evaluation at psi^(2i+1)."""
    if not (q == Q and a.size == N and b.size == N):
        raise ValueError("q == Q and a.size == N and b.size == N")
    tw = np.array([_pow(PSI, i) for i in range(N)])
    twi = np.array([_pow(PSI, -i) for i in range(N)])
    ah = ntt((a % Q) * tw % Q)
    bh = ntt((b % Q) * tw % Q)
    return np.asarray(intt(ah * bh % Q) * twi % Q)


def _schoolbook(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.zeros(N, dtype=np.int64)
    for i in range(N):
        for j in range(N):
            s = 1 if i + j < N else -1
            c[(i + j) % N] = (c[(i + j) % N] + s * int(a[i]) * int(b[j])) % Q
    return c


def bench_ntt_ring(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: NTT product == schoolbook; inverse transform round-trip."""
    rng = np.random.default_rng(seed)
    trials = 10
    ok = True
    for _ in range(trials):
        a = rng.integers(0, Q, N)
        b = rng.integers(0, Q, N)
        ok &= bool(np.array_equal(ntt_mul(a, b) % Q, _schoolbook(a, b) % Q))
        ok &= bool(np.array_equal(intt(ntt(a % Q)), a % Q))
    return {"synthetic_ntt_ring": 1.0 if ok else 0.0}
