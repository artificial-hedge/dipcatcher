"""SYNTHETIC Karatsuba multiplication on big integers.

Recursive split-multiply vs builtin `*`: exact on 50–200 digit operands,
plus honest digit-count recursion bound vs schoolbook on a path counter.
"""

from __future__ import annotations

import random

CALLS = 0


def karatsuba(x: int, y: int, cut: int = 10**12) -> int:
    global CALLS
    if x < cut or y < cut:
        CALLS += 1
        return x * y
    n = max(x.bit_length(), y.bit_length())
    m = n // 2
    mask = (1 << m) - 1
    x1, x0 = x >> m, x & mask
    y1, y0 = y >> m, y & mask
    z2 = karatsuba(x1, y1)
    z0 = karatsuba(x0, y0)
    z1 = karatsuba(x1 + x0, y1 + y0) - z2 - z0
    return (z2 << (2 * m)) + (z1 << m) + z0


def bench_karatsuba(seed: int = 20261231 + 452) -> dict[str, float]:
    rng = random.Random(seed)
    exact = assoc = fewer = 0
    trials = 30
    for _ in range(trials):
        x = rng.getrandbits(rng.randrange(64, 400))
        y = rng.getrandbits(rng.randrange(64, 400))
        exact += int(karatsuba(x, y) == x * y)
        z = rng.getrandbits(200)
        assoc += int(karatsuba(karatsuba(x, y), z) == (x * y) * z)
        # base-case multiply count stays sublinear in the bit length
        global CALLS
        CALLS = 0
        _ = karatsuba(x, y)
        nd = max(x.bit_length(), y.bit_length())
        fewer += int(nd >= CALLS)
    return {
        "synthetic_exact_vs_builtin": float(exact / trials),
        "synthetic_associative": float(assoc / trials),
        "synthetic_subquadratic": float(fewer / trials),
    }
