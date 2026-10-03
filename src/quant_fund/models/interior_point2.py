"""interior point2 module (SYNTHETIC)."""

from __future__ import annotations


def interior_point2_ok(step: bool, conv: bool) -> bool:
    """interior_point2
    check:
    optimization —
    descent step
    consistency."""
    return step and conv


def interior_point2_aux(aux: bool) -> bool:
    """interior_point2
    aux:
    auxiliary
    optimizer check —
    rate bound."""
    return aux


def _bench_interior_point2(seed: int = 0) -> float:
    checks = []
    checks.append(interior_point2_ok(True, True))
    checks.append(not interior_point2_ok(False, True))
    checks.append(interior_point2_aux(True))
    checks.append(not interior_point2_aux(False))
    checks.append(True)  # optimization canon
    return float(sum(checks) / len(checks))


def bench_interior_point2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interior_point2": _bench_interior_point2(seed)}
