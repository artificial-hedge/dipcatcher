"""derived separated2 module (SYNTHETIC)."""

from __future__ import annotations


def derived_separated2_ok(motivic: bool, categorical: bool) -> bool:
    """derived_separated2
    check:
    motivic
    structure —
    functor."""
    return motivic and categorical


def derived_separated2_aux(aux: bool) -> bool:
    """derived_separated2
    aux:
    auxiliary
    motivic
    check —
    nerve."""
    return aux


def _bench_derived_separated2(seed: int = 0) -> float:
    checks = []
    checks.append(derived_separated2_ok(True, True))
    checks.append(not derived_separated2_ok(False, True))
    checks.append(derived_separated2_aux(True))
    checks.append(not derived_separated2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_derived_separated2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_separated2": _bench_derived_separated2(seed)}
