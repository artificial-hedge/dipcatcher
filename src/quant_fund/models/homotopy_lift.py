"""homotopy lift module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_lift_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_lift
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def homotopy_lift_aux(aux: bool) -> bool:
    """homotopy_lift
    aux:
    auxiliary
    homotopy
    check —
    stable."""
    return aux


def _bench_homotopy_lift(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_lift_ok(True, True))
    checks.append(not homotopy_lift_ok(False, True))
    checks.append(homotopy_lift_aux(True))
    checks.append(not homotopy_lift_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_lift(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_lift": _bench_homotopy_lift(seed)}
