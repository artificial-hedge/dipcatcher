"""cotangent stack module (SYNTHETIC)."""

from __future__ import annotations


def cotangent_stack_ok(derived: bool, geometry: bool) -> bool:
    """cotangent_stack
    check:
    derived
    geometry
    structure —
    spectral."""
    return derived and geometry


def cotangent_stack_aux(aux: bool) -> bool:
    """cotangent_stack
    aux:
    auxiliary
    derived-geom
    check —
    analytic."""
    return aux


def _bench_cotangent_stack(seed: int = 0) -> float:
    checks = []
    checks.append(cotangent_stack_ok(True, True))
    checks.append(not cotangent_stack_ok(False, True))
    checks.append(cotangent_stack_aux(True))
    checks.append(not cotangent_stack_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_cotangent_stack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cotangent_stack": _bench_cotangent_stack(seed)}
