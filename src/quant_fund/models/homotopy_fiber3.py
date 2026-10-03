"""homotopy fiber3 module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_fiber3_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_fiber3
    check:
    homotopy
    structure —
    suspension."""
    return homotopy and stable


def homotopy_fiber3_aux(aux: bool) -> bool:
    """homotopy_fiber3
    aux:
    auxiliary
    homotopy
    check —
    fiber."""
    return aux


def _bench_homotopy_fiber3(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_fiber3_ok(True, True))
    checks.append(not homotopy_fiber3_ok(False, True))
    checks.append(homotopy_fiber3_aux(True))
    checks.append(not homotopy_fiber3_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_fiber3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_fiber3": _bench_homotopy_fiber3(seed)}
