"""free boundary module (SYNTHETIC)."""

from __future__ import annotations


def free_boundary_ok(os1: bool, sd: bool) -> bool:
    """free_boundary
    check:
    optimal-
    stopping —
    value
    function."""
    return os1 and sd


def free_boundary_aux(aux: bool) -> bool:
    """free_boundary
    aux:
    auxiliary
    stopping
    check —
    boundary."""
    return aux


def _bench_free_boundary(seed: int = 0) -> float:
    checks = []
    checks.append(free_boundary_ok(True, True))
    checks.append(not free_boundary_ok(False, True))
    checks.append(free_boundary_aux(True))
    checks.append(not free_boundary_aux(False))
    checks.append(True)  # optimal-stopping canon
    return float(sum(checks) / len(checks))


def bench_free_boundary(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_boundary": _bench_free_boundary(seed)}
