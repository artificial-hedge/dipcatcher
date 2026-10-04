"""Inclusion-exclusion: derangements and surjections (SYNTHETIC)."""

from __future__ import annotations

import itertools
import math


def derangement(n: int) -> int:
    """D_n = n! * sum_{k=0}^n (-1)^k / k!"""
    return int(round(math.factorial(n) * sum((-1) ** k / math.factorial(k) for k in range(n + 1))))


def surjections(n: int, m: int) -> int:
    """|surj(n -> m)| = sum_{k=0}^m (-1)^k C(m,k) (m-k)^n."""
    return sum((-1) ** k * math.comb(m, k) * (m - k) ** n for k in range(m + 1))


def _bench_inclusion_excl(seed: int = 0) -> float:
    checks = []
    # derangement numbers D_1..D_5 = 0,1,2,9,44
    checks.append([derangement(n) for n in range(1, 6)] == [0, 1, 2, 9, 44])
    # brute-force derangements of 4
    cnt = sum(1 for p in itertools.permutations(range(4)) if all(p[i] != i for i in range(4)))
    checks.append(cnt == 9)
    # surjections onto 2 of 3 elements = 6
    checks.append(surjections(3, 2) == 6)
    # surjections n -> n = n!
    checks.append(surjections(4, 4) == 24)
    # surjections 4 -> 2 = 14
    checks.append(surjections(4, 2) == 14)
    return float(sum(checks) / len(checks))


def bench_inclusion_excl(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inclusion_excl": _bench_inclusion_excl(seed)}
