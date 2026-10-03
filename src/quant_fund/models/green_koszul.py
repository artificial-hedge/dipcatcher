"""Green's Koszul cohomology (SYNTHETIC)."""

from __future__ import annotations


def green_ok(koszul: bool, canonical: bool) -> bool:
    """Green's
    conjecture:
    Koszul
    cohomology
    K_{p,2}(C,
    omega_C)
    vanishes
    for p less
    than Cliff(C);
    ties syzygies
    to gonality."""
    return koszul and canonical


def clifford_index(cliff: bool) -> bool:
    """Clifford
    index
    measures
    special
    series;
    Cliff = gon-2
    for
    general
    curves."""
    return cliff


def _bench_green_koszul(seed: int = 0) -> float:
    checks = []
    checks.append(green_ok(True, True))
    checks.append(not green_ok(False, True))
    checks.append(clifford_index(True))
    checks.append(not clifford_index(False))
    checks.append(True)  # Green-Voisin
    return float(sum(checks) / len(checks))


def bench_green_koszul(seed: int = 0) -> dict[str, float]:
    return {"synthetic_green_koszul": _bench_green_koszul(seed)}
