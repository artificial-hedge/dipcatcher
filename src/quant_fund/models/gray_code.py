"""Binary-reflected Gray code construction + validation (wave 282).

g(i) = i ^ (i >> 1): consecutive codewords differ in exactly one bit, the
sequence is a permutation of 0..2^n-1, and it wraps around cyclically.
"""

import numpy as np

_SEED = 20261231 + 778


def gray(n: int) -> list[int]:
    return [i ^ (i >> 1) for i in range(1 << n)]


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def bench_gray_code(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(4):
        n = int(rng.randint(2, 8))
        g = gray(n)
        adj = all(_hamming(g[i], g[i + 1]) == 1 for i in range(len(g) - 1))
        cyc = _hamming(g[0], g[-1]) == 1
        ok += int(adj and cyc and len(set(g)) == 1 << n and set(g) == set(range(1 << n)))
    return {"synthetic_gray_valid": float(ok == 4)}
