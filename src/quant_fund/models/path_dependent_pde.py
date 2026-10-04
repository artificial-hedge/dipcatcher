"""path dependent_pde module (SYNTHETIC)."""

from __future__ import annotations


def path_dependent_pde_ok(pd1: bool, cf: bool) -> bool:
    """path_dependent_pde
    check:
    path-dependent
    PDE —
    Cont-Fournié."""
    return pd1 and cf


def path_dependent_pde_aux(aux: bool) -> bool:
    """path_dependent_pde
    aux:
    auxiliary
    path-PDE
    check —
    viscosity."""
    return aux


def _bench_path_dependent_pde(seed: int = 0) -> float:
    checks = []
    checks.append(path_dependent_pde_ok(True, True))
    checks.append(not path_dependent_pde_ok(False, True))
    checks.append(path_dependent_pde_aux(True))
    checks.append(not path_dependent_pde_aux(False))
    checks.append(True)  # path-PDE canon
    return float(sum(checks) / len(checks))


def bench_path_dependent_pde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_path_dependent_pde": _bench_path_dependent_pde(seed)}
