"""Unsigned Stirling numbers of the first kind (SYNTHETIC)."""

from __future__ import annotations

import itertools


def stirling1(n: int, k: int) -> int:
    """[n k] via recurrence [n k] = (n-1)[n-1 k] + [n-1 k-1]."""
    if n == 0:
        return 1 if k == 0 else 0
    if k == 0:
        return 0
    return (n - 1) * stirling1(n - 1, k) + stirling1(n - 1, k - 1)


def count_by_cycles(n: int, k: int) -> int:
    """Permutations of n with exactly k cycles (brute force)."""
    cnt = 0
    for perm in itertools.permutations(range(n)):
        seen = [False] * n
        cyc = 0
        for i in range(n):
            if not seen[i]:
                cyc += 1
                j = i
                while not seen[j]:
                    seen[j] = True
                    j = perm[j]
        if cyc == k:
            cnt += 1
    return cnt


def _bench_stirling_cycle(seed: int = 0) -> float:
    checks = []
    # row n=4 is [6, 11, 6, 1] for k = 1..4
    checks.append([stirling1(4, k) for k in range(1, 5)] == [6, 11, 6, 1])
    # row sum = n!
    import math

    checks.append(sum(stirling1(4, k) for k in range(1, 5)) == math.factorial(4))
    # recurrence sanity
    checks.append(stirling1(5, 2) == 4 * stirling1(4, 2) + stirling1(4, 1))
    # brute-force agreement at n=4
    checks.append(all(count_by_cycles(4, k) == stirling1(4, k) for k in range(1, 5)))
    return float(sum(checks) / len(checks))


def bench_stirling_cycle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stirling_cycle": _bench_stirling_cycle(seed)}
