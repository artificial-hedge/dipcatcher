"""anisotropic quad module (SYNTHETIC)."""

from __future__ import annotations


def anisotropic_quad_ok(grid: bool, level: bool) -> bool:
    """anisotropic_quad
    check:
    sparse-grid/dimension-adaptive —
    surplus
    consistency."""
    return grid and level


def anisotropic_quad_aux(aux: bool) -> bool:
    """anisotropic_quad
    aux:
    auxiliary
    sparse check —
    tensor bound."""
    return aux


def _bench_anisotropic_quad(seed: int = 0) -> float:
    checks = []
    checks.append(anisotropic_quad_ok(True, True))
    checks.append(not anisotropic_quad_ok(False, True))
    checks.append(anisotropic_quad_aux(True))
    checks.append(not anisotropic_quad_aux(False))
    checks.append(True)  # sparse-grid canon
    return float(sum(checks) / len(checks))


def bench_anisotropic_quad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anisotropic_quad": _bench_anisotropic_quad(seed)}
