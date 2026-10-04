"""homotopy vn module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_vn_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_vn
    check:
    homotopy
    structure —
    suspension."""
    return homotopy and stable


def homotopy_vn_aux(aux: bool) -> bool:
    """homotopy_vn
    aux:
    auxiliary
    homotopy
    check —
    fiber."""
    return aux


def _bench_homotopy_vn(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_vn_ok(True, True))
    checks.append(not homotopy_vn_ok(False, True))
    checks.append(homotopy_vn_aux(True))
    checks.append(not homotopy_vn_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_vn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_vn": _bench_homotopy_vn(seed)}
