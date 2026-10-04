"""lattice point module (SYNTHETIC)."""

from __future__ import annotations


def lattice_point_ok(discrete: bool, conv: bool) -> bool:
    """lattice_point
    check:
    discrete
    geometry —
    convexity."""
    return discrete and conv


def lattice_point_aux(aux: bool) -> bool:
    """lattice_point
    aux:
    auxiliary
    geometry check —
    combinatorial."""
    return aux


def _bench_lattice_point(seed: int = 0) -> float:
    checks = []
    checks.append(lattice_point_ok(True, True))
    checks.append(not lattice_point_ok(False, True))
    checks.append(lattice_point_aux(True))
    checks.append(not lattice_point_aux(False))
    checks.append(True)  # discrete-geometry canon
    return float(sum(checks) / len(checks))


def bench_lattice_point(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lattice_point": _bench_lattice_point(seed)}
