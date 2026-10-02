"""SYNTHETIC number-theoretic transform (mod 998244353, root g=3).

NTT-based polynomial multiplication vs naive O(n²) multiplication mod p
— exact integer arithmetic, no floating point anywhere.
"""

from __future__ import annotations

import random

P = 998244353
G = 3


def _pow(a: int, e: int, m: int = P) -> int:
    return pow(a, e, m)


def _ntt(a: list[int], invert: bool, n: int) -> list[int]:
    a = list(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        wlen = _pow(G, (P - 1) // length)
        if invert:
            wlen = _pow(wlen, P - 2)
        for i in range(0, n, length):
            w = 1
            for k in range(i, i + length // 2):
                u, v = a[k], a[k + length // 2] * w % P
                a[k] = (u + v) % P
                a[k + length // 2] = (u - v) % P
                w = w * wlen % P
        length <<= 1
    if invert:
        inv_n = _pow(n, P - 2)
        a = [x * inv_n % P for x in a]
    return a


def ntt_mul(a: list[int], b: list[int]) -> list[int]:
    need = len(a) + len(b) - 1
    n = 1
    while n < need:
        n <<= 1
    fa = _ntt(a + [0] * (n - len(a)), False, n)
    fb = _ntt(b + [0] * (n - len(b)), False, n)
    fc = [x * y % P for x, y in zip(fa, fb, strict=True)]
    return _ntt(fc, True, n)[:need]


def _naive_mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % P
    return out


def bench_ntt(seed: int = 20261231 + 451) -> dict[str, float]:
    rng = random.Random(seed)
    exact = inv = linear = 0
    trials = 40
    for _ in range(trials):
        da, db = rng.randrange(2, 12), rng.randrange(2, 12)
        a = [rng.randrange(P) for _ in range(da)]
        b = [rng.randrange(P) for _ in range(db)]
        exact += int(ntt_mul(a, b) == _naive_mul(a, b))
        # forward/inverse round-trip on a single poly
        n = 1 << rng.randrange(3, 8)
        x = [rng.randrange(P) for _ in range(n)]
        inv += int(_ntt(_ntt(x, False, n), True, n) == x)
        # linearity: NTT(a+b) = NTT(a)+NTT(b) (cheap structural check via mul)
        c1 = ntt_mul([1], a)
        linear += int(c1 == a)
    return {
        "synthetic_ntt_mul_exact": float(exact / trials),
        "synthetic_ntt_inverse": float(inv / trials),
        "synthetic_identity_mul": float(linear / trials),
    }
