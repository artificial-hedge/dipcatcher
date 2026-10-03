"""SYNTHETIC Newton integer square root + k-th root.

isqrt(n) via Newton's method, verified against math.isqrt on 10–300 bit
inputs; kth_root(n,k) floor-verified by (r+1)^k > n >= r^k.
"""

from __future__ import annotations

import math
import random


def isqrt(n: int) -> int:
    if n < 2:
        return n
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def kth_root(n: int, k: int) -> int:
    if n < 2 or k == 1:
        return n
    lo, hi = 1, 1 << (n.bit_length() // k + 2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**k <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def bench_int_sqrt(seed: int = 20261231 + 454) -> dict[str, float]:
    rng = random.Random(seed)
    exact = floor = kth = 0
    trials = 60
    for _ in range(trials):
        n = rng.getrandbits(rng.randrange(10, 300))
        r = isqrt(n)
        exact += int(r == math.isqrt(n))
        floor += int(r * r <= n < (r + 1) * (r + 1))
        k = rng.randrange(2, 8)
        rk = kth_root(n, k)
        kth += int(rk**k <= n < (rk + 1) ** k)
    return {
        "synthetic_isqrt_exact": float(exact / trials),
        "synthetic_floor_property": float(floor / trials),
        "synthetic_kth_root_floor": float(kth / trials),
    }
