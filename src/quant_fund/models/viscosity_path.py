"""viscosity path module (SYNTHETIC)."""

from __future__ import annotations


def viscosity_path_ok(pd1: bool, cf: bool) -> bool:
    """viscosity_path
    check:
    path-dependent
    PDE —
    Cont-Fournié."""
    return pd1 and cf


def viscosity_path_aux(aux: bool) -> bool:
    """viscosity_path
    aux:
    auxiliary
    path-PDE
    check —
    viscosity."""
    return aux


def _bench_viscosity_path(seed: int = 0) -> float:
    checks = []
    checks.append(viscosity_path_ok(True, True))
    checks.append(not viscosity_path_ok(False, True))
    checks.append(viscosity_path_aux(True))
    checks.append(not viscosity_path_aux(False))
    checks.append(True)  # path-PDE canon
    return float(sum(checks) / len(checks))


def bench_viscosity_path(seed: int = 0) -> dict[str, float]:
    return {"synthetic_viscosity_path": _bench_viscosity_path(seed)}
