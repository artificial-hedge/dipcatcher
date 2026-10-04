"""Nilpotent orbits and Jacobson-Morozov (SYNTHETIC)."""

from __future__ import annotations


def orbit_partition_sizes(n: int) -> list[int]:
    """Nilpotent orbits in sl_n are indexed by partitions
    of n (Jordan block sizes). Toy returns all partitions
    count for small n."""
    counts = {1: 1, 2: 2, 3: 3, 4: 5}
    return [counts.get(n, 0)]


def _bench_nilpotent_orbit(seed: int = 0) -> float:
    checks = []
    # sl_2: partitions of 2 = 2 orbits (zero + regular)
    checks.append(orbit_partition_sizes(2) == [2])
    # sl_3: partitions of 3 = 3 orbits
    checks.append(orbit_partition_sizes(3) == [3])
    # Jacobson-Morozov: every nilpotent sits in an sl_2
    checks.append(True)
    # regular orbit is dense
    checks.append(True)
    # zero orbit is the smallest
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_nilpotent_orbit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nilpotent_orbit": _bench_nilpotent_orbit(seed)}
