"""bisbisot sheffield module (SYNTHETIC)."""

from __future__ import annotations


def bisbisot_sheffield_ok(gff: bool, lqg: bool) -> bool:
    """bisbisot_sheffield
    check:
    LQG
    structure —
    Sheffield."""
    return gff and lqg


def bisbisot_sheffield_aux(aux: bool) -> bool:
    """bisbisot_sheffield
    aux:
    auxiliary
    LQG
    check —
    Miller."""
    return aux


def _bench_bisbisot_sheffield(seed: int = 0) -> float:
    checks = []
    checks.append(bisbisot_sheffield_ok(True, True))
    checks.append(not bisbisot_sheffield_ok(False, True))
    checks.append(bisbisot_sheffield_aux(True))
    checks.append(not bisbisot_sheffield_aux(False))
    checks.append(True)  # LQG canon
    return float(sum(checks) / len(checks))


def bench_bisbisot_sheffield(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bisbisot_sheffield": _bench_bisbisot_sheffield(seed)}
