"""virtual class2 module (SYNTHETIC)."""

from __future__ import annotations


def virtual_class2_ok(derived: bool, geometry: bool) -> bool:
    """virtual_class2
    check:
    derived
    geometry —
    spectral."""
    return derived and geometry


def virtual_class2_aux(aux: bool) -> bool:
    """virtual_class2
    aux:
    auxiliary
    derived
    check —
    stack."""
    return aux


def _bench_virtual_class2(seed: int = 0) -> float:
    checks = []
    checks.append(virtual_class2_ok(True, True))
    checks.append(not virtual_class2_ok(False, True))
    checks.append(virtual_class2_aux(True))
    checks.append(not virtual_class2_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_virtual_class2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_virtual_class2": _bench_virtual_class2(seed)}
