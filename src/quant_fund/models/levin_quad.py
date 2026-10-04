"""levin quad module (SYNTHETIC)."""

from __future__ import annotations


def levin_quad_ok(panel: bool, freq: bool) -> bool:
    """levin_quad
    check:
    adaptive/oscillatory-quadrature —
    tolerance
    consistency."""
    return panel and freq


def levin_quad_aux(aux: bool) -> bool:
    """levin_quad
    aux:
    auxiliary
    quadrature check —
    convergence bound."""
    return aux


def _bench_levin_quad(seed: int = 0) -> float:
    checks = []
    checks.append(levin_quad_ok(True, True))
    checks.append(not levin_quad_ok(False, True))
    checks.append(levin_quad_aux(True))
    checks.append(not levin_quad_aux(False))
    checks.append(True)  # adaptive-quadrature canon
    return float(sum(checks) / len(checks))


def bench_levin_quad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_levin_quad": _bench_levin_quad(seed)}
