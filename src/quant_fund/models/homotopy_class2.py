"""homotopy class2 module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_class2_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_class2
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def homotopy_class2_aux(aux: bool) -> bool:
    """homotopy_class2
    aux:
    auxiliary
    homotopy
    check —
    limit."""
    return aux


def _bench_homotopy_class2(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_class2_ok(True, True))
    checks.append(not homotopy_class2_ok(False, True))
    checks.append(homotopy_class2_aux(True))
    checks.append(not homotopy_class2_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_class2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_class2": _bench_homotopy_class2(seed)}
