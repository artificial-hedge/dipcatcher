"""Latin square counts and isotopy (SYNTHETIC)."""

from __future__ import annotations

import math


def latin_count(n: int) -> int:
    """Number of Latin squares of order n (OEIS A000315 for reduced,
    times n!(n-1)! for full count)."""
    reduced = {1: 1, 2: 1, 3: 1, 4: 4, 5: 56}
    r = reduced[n]
    return r * math.factorial(n) * math.factorial(n - 1)


def reduced_count(n: int) -> int:
    return {1: 1, 2: 1, 3: 1, 4: 4, 5: 56}[n]


def _bench_latin_trade(seed: int = 0) -> float:
    checks = []
    # N(3) = 12
    checks.append(latin_count(3) == 12)
    # N(4) = 576
    checks.append(latin_count(4) == 576)
    # reduced counts: 1,1,1,4,56
    checks.append(reduced_count(4) == 4)
    checks.append(reduced_count(5) == 56)
    # N(2) = 2
    checks.append(latin_count(2) == 2)
    # latin_count(n) = reduced * n! (n-1)!
    checks.append(latin_count(5) == reduced_count(5) * math.factorial(5) * math.factorial(4))
    return float(sum(checks) / len(checks))


def bench_latin_trade(seed: int = 0) -> dict[str, float]:
    return {"synthetic_latin_trade": _bench_latin_trade(seed)}
