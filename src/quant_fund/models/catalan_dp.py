"""Catalan numbers and bijective manifestations (SYNTHETIC)."""

from __future__ import annotations

import itertools


def catalan(n: int) -> int:
    c = [1] + [0] * n
    for i in range(1, n + 1):
        c[i] = sum(c[k] * c[i - 1 - k] for k in range(i))
    return c[n]


def valid_parens(n: int) -> int:
    """Count balanced strings of 2n parentheses by exhaustive check."""
    total = 0
    for bits in itertools.combinations(range(2 * n), n):
        opens = set(bits)
        depth = 0
        ok = True
        for i in range(2 * n):
            depth += 1 if i in opens else -1
            if depth < 0:
                ok = False
                break
        if ok and depth == 0:
            total += 1
    return total


def _bench_catalan_dp(seed: int = 0) -> float:
    checks = []
    checks.append([catalan(n) for n in range(6)] == [1, 1, 2, 5, 14, 42])
    # Dyck-path / paren bijection: valid strings of 2n parens = C_n
    checks.append(valid_parens(3) == 5)
    checks.append(valid_parens(4) == 14)
    # closed form C_n = (1/(n+1)) * C(2n, n)
    import math

    checks.append(catalan(5) == math.comb(10, 5) // 6)
    checks.append(catalan(4) == math.comb(8, 4) // 5)
    return float(sum(checks) / len(checks))


def bench_catalan_dp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_catalan_dp": _bench_catalan_dp(seed)}
