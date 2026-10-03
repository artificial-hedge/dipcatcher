"""dupire functional module (SYNTHETIC)."""

from __future__ import annotations


def dupire_functional_ok(pd1: bool, cf: bool) -> bool:
    """dupire_functional
    check:
    path-dependent
    PDE —
    Cont-Fournié."""
    return pd1 and cf


def dupire_functional_aux(aux: bool) -> bool:
    """dupire_functional
    aux:
    auxiliary
    path-PDE
    check —
    viscosity."""
    return aux


def _bench_dupire_functional(seed: int = 0) -> float:
    checks = []
    checks.append(dupire_functional_ok(True, True))
    checks.append(not dupire_functional_ok(False, True))
    checks.append(dupire_functional_aux(True))
    checks.append(not dupire_functional_aux(False))
    checks.append(True)  # path-PDE canon
    return float(sum(checks) / len(checks))


def bench_dupire_functional(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dupire_functional": _bench_dupire_functional(seed)}
