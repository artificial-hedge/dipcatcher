"""Eulerian numbers A(n, m) (SYNTHETIC)."""

from __future__ import annotations

import itertools


def eulerian(n: int, m: int) -> int:
    """A(n,m) = permutations of n with exactly m descents."""
    if n == 0:
        return 1 if m == 0 else 0
    if m < 0 or m >= n:
        return 0
    return (m + 1) * eulerian(n - 1, m) + (n - m) * eulerian(n - 1, m - 1)


def descents(perm: tuple[int, ...]) -> int:
    return sum(1 for i in range(len(perm) - 1) if perm[i] > perm[i + 1])


def _bench_eulerian_num(seed: int = 0) -> float:
    checks = []
    # row n=3: [1, 4, 1]
    checks.append([eulerian(3, m) for m in range(3)] == [1, 4, 1])
    # row n=4: [1, 11, 11, 1]
    checks.append([eulerian(4, m) for m in range(4)] == [1, 11, 11, 1])
    # row sum = n!
    import math

    checks.append(sum(eulerian(4, m) for m in range(4)) == math.factorial(4))
    # brute force at n=4
    cnt = [0] * 4
    for p in itertools.permutations(range(4)):
        cnt[descents(p)] += 1
    checks.append(cnt == [1, 11, 11, 1])
    # symmetry A(n,m) = A(n, n-1-m)
    checks.append(all(eulerian(4, m) == eulerian(4, 3 - m) for m in range(4)))
    return float(sum(checks) / len(checks))


def bench_eulerian_num(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eulerian_num": _bench_eulerian_num(seed)}
