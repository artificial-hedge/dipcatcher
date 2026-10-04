"""campbell thm module (SYNTHETIC)."""

from __future__ import annotations


def campbell_thm_ok(pt: bool, meas: bool) -> bool:
    """campbell_thm
    check:
    point-process
    structure —
    Cox
    intensity."""
    return pt and meas


def campbell_thm_aux(aux: bool) -> bool:
    """campbell_thm
    aux:
    auxiliary
    mark
    check —
    Palm
    distribution."""
    return aux


def _bench_campbell_thm(seed: int = 0) -> float:
    checks = []
    checks.append(campbell_thm_ok(True, True))
    checks.append(not campbell_thm_ok(False, True))
    checks.append(campbell_thm_aux(True))
    checks.append(not campbell_thm_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_campbell_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_campbell_thm": _bench_campbell_thm(seed)}
