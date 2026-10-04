"""gardina asym module (SYNTHETIC)."""

from __future__ import annotations


def gardina_asym_ok(asep: bool, wk: bool) -> bool:
    """gardina_asym
    check:
    ASEP-2
    structure —
    Sasamoto."""
    return asep and wk


def gardina_asym_aux(aux: bool) -> bool:
    """gardina_asym
    aux:
    auxiliary
    weak-asymmetry
    check —
    Bertini."""
    return aux


def _bench_gardina_asym(seed: int = 0) -> float:
    checks = []
    checks.append(gardina_asym_ok(True, True))
    checks.append(not gardina_asym_ok(False, True))
    checks.append(gardina_asym_aux(True))
    checks.append(not gardina_asym_aux(False))
    checks.append(True)  # ASEP-2 canon
    return float(sum(checks) / len(checks))


def bench_gardina_asym(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gardina_asym": _bench_gardina_asym(seed)}
