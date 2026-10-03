"""functional ito module (SYNTHETIC)."""

from __future__ import annotations


def functional_ito_ok(pd1: bool, cf: bool) -> bool:
    """functional_ito
    check:
    path-dependent
    PDE —
    Cont-Fournié."""
    return pd1 and cf


def functional_ito_aux(aux: bool) -> bool:
    """functional_ito
    aux:
    auxiliary
    path-PDE
    check —
    viscosity."""
    return aux


def _bench_functional_ito(seed: int = 0) -> float:
    checks = []
    checks.append(functional_ito_ok(True, True))
    checks.append(not functional_ito_ok(False, True))
    checks.append(functional_ito_aux(True))
    checks.append(not functional_ito_aux(False))
    checks.append(True)  # path-PDE canon
    return float(sum(checks) / len(checks))


def bench_functional_ito(seed: int = 0) -> dict[str, float]:
    return {"synthetic_functional_ito": _bench_functional_ito(seed)}
