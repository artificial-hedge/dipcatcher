"""PID check on Z: every ideal principal via Euclidean gcd (SYNTHETIC)."""

from __future__ import annotations


def euclid_gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def bezout(a: int, b: int) -> tuple[int, int, int]:
    """(g, x, y) with g = ax + by."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = bezout(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def ideal_gen(elems: frozenset[int]) -> int:
    """Smallest nonneg generator = gcd of all elements."""
    g = 0
    for x in elems:
        g = euclid_gcd(g, x)
    return g


def _bench_pid_check(seed: int = 0) -> float:
    checks = []
    checks.append(euclid_gcd(12, 18) == 6)
    checks.append(euclid_gcd(0, 5) == 5)
    g, x, y = bezout(6, 9)
    checks.append(g == 3 and 6 * x + 9 * y == 3)
    checks.append(ideal_gen(frozenset({4, 6, 10})) == 2)
    checks.append(ideal_gen(frozenset({7})) == 7)
    checks.append(ideal_gen(frozenset({0})) == 0)
    return float(sum(checks) / len(checks))


def bench_pid_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pid_check": _bench_pid_check(seed)}
