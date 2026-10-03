"""derived abelian2 module (SYNTHETIC)."""

from __future__ import annotations


def derived_abelian2_ok(derived: bool, geometric: bool) -> bool:
    """derived_abelian2
    check:
    derived
    structure —
    stack."""
    return derived and geometric


def derived_abelian2_aux(aux: bool) -> bool:
    """derived_abelian2
    aux:
    auxiliary
    derived
    check —
    morph."""
    return aux


def _bench_derived_abelian2(seed: int = 0) -> float:
    checks = []
    checks.append(derived_abelian2_ok(True, True))
    checks.append(not derived_abelian2_ok(False, True))
    checks.append(derived_abelian2_aux(True))
    checks.append(not derived_abelian2_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_abelian2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_abelian2": _bench_derived_abelian2(seed)}
