"""Bell numbers via Stirling second kind (SYNTHETIC)."""

from __future__ import annotations


def s2(n: int, k: int) -> int:
    """Stirling {n k}: partitions of n-set into k blocks."""
    if n == 0:
        return 1 if k == 0 else 0
    if k == 0 or k > n:
        return 0
    return k * s2(n - 1, k) + s2(n - 1, k - 1)


def bell(n: int) -> int:
    return sum(s2(n, k) for k in range(n + 1))


def _bench_bell_triangle(seed: int = 0) -> float:
    checks = []
    checks.append([bell(n) for n in range(6)] == [1, 1, 2, 5, 15, 52])
    # {4 2} = 7
    checks.append(s2(4, 2) == 7)
    # {n n} = 1, {n 1} = 1
    checks.append(s2(5, 5) == 1 and s2(5, 1) == 1)
    # {n 2} = 2^{n-1} - 1
    checks.append(s2(5, 2) == 15)
    # recurrence {n k} = k{n-1 k} + {n-1 k-1}
    checks.append(s2(6, 3) == 3 * s2(5, 3) + s2(5, 2))
    return float(sum(checks) / len(checks))


def bench_bell_triangle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bell_triangle": _bench_bell_triangle(seed)}
