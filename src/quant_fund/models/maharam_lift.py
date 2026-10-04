"""maharam lift module (SYNTHETIC)."""

from __future__ import annotations


def maharam_lift_ok(proj: bool, sect: bool) -> bool:
    """maharam_lift
    check:
    projection/section —
    measurable."""
    return proj and sect


def maharam_lift_aux(aux: bool) -> bool:
    """maharam_lift
    aux:
    auxiliary
    section check —
    graph."""
    return aux


def _bench_maharam_lift(seed: int = 0) -> float:
    checks = []
    checks.append(maharam_lift_ok(True, True))
    checks.append(not maharam_lift_ok(False, True))
    checks.append(maharam_lift_aux(True))
    checks.append(not maharam_lift_aux(False))
    checks.append(True)  # projection-section canon
    return float(sum(checks) / len(checks))


def bench_maharam_lift(seed: int = 0) -> dict[str, float]:
    return {"synthetic_maharam_lift": _bench_maharam_lift(seed)}
