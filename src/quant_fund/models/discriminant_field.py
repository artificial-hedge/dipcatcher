"""Discriminant of Q(sqrt(d)) for squarefree d (SYNTHETIC)."""

from __future__ import annotations


def quad_disc(d: int) -> int:
    """disc Q(sqrt(d)) = d if d == 1 mod 4, else 4d."""
    return d if d % 4 == 1 else 4 * d


def ramified_primes(d: int, primes: list[int]) -> list[int]:
    """Ramified primes are exactly those dividing the discriminant."""
    return [p for p in primes if quad_disc(d) % p == 0]


def _bench_discriminant_field(seed: int = 0) -> float:
    checks = []
    # disc Q(i) = -4
    checks.append(quad_disc(-1) == -4)
    # disc Q(sqrt(2)) = 8
    checks.append(quad_disc(2) == 8)
    # disc Q(sqrt(5)) = 5 (d == 1 mod 4)
    checks.append(quad_disc(5) == 5)
    # disc Q(sqrt(-3)) = -3
    checks.append(quad_disc(-3) == -3)
    # ramified primes of Q(sqrt(3)): D = 12 -> {2,3}
    checks.append(ramified_primes(3, [2, 3, 5, 7]) == [2, 3])
    # only 5 ramifies in Q(sqrt(5))
    checks.append(ramified_primes(5, [2, 3, 5, 7]) == [5])
    return float(sum(checks) / len(checks))


def bench_discriminant_field(seed: int = 0) -> dict[str, float]:
    return {"synthetic_discriminant_field": _bench_discriminant_field(seed)}
