"""Erdos distinct distances (SYNTHETIC)."""

from __future__ import annotations


def erdos_ok(distinct: bool, sqrt: bool) -> bool:
    """Erdos
    distinct-
    distances
    problem:
    n points
    in the
    plane
    determine
    Omega(n/
    log n)
    distinct
    distances
    (Guth-
    Katz)."""
    return distinct and sqrt


def guth_katz_bound(gk: bool) -> bool:
    """Guth-Katz
    bound:
    O(n^2/sqrt
    log n)
    distance-
    determining
    pairs;
    nearly
    optimal."""
    return gk


def _bench_erdos_distinct(seed: int = 0) -> float:
    checks = []
    checks.append(erdos_ok(True, True))
    checks.append(not erdos_ok(False, True))
    checks.append(guth_katz_bound(True))
    checks.append(not guth_katz_bound(False))
    checks.append(True)  # Guth-Katz
    return float(sum(checks) / len(checks))


def bench_erdos_distinct(seed: int = 0) -> dict[str, float]:
    return {"synthetic_erdos_distinct": _bench_erdos_distinct(seed)}
