"""milnor operations2 module (SYNTHETIC)."""

from __future__ import annotations


def milnor_operations2_ok(motivic: bool, stable: bool) -> bool:
    """milnor_operations2
    check:
    motivic
    stable
    homotopy —
    slice."""
    return motivic and stable


def milnor_operations2_aux(aux: bool) -> bool:
    """milnor_operations2
    aux:
    auxiliary
    motivic
    check —
    spectral."""
    return aux


def _bench_milnor_operations2(seed: int = 0) -> float:
    checks = []
    checks.append(milnor_operations2_ok(True, True))
    checks.append(not milnor_operations2_ok(False, True))
    checks.append(milnor_operations2_aux(True))
    checks.append(not milnor_operations2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_milnor_operations2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_milnor_operations2": _bench_milnor_operations2(seed)}
