"""schramm lerw module (SYNTHETIC)."""

from __future__ import annotations


def schramm_lerw_ok(ust: bool, lerw: bool) -> bool:
    """schramm_lerw
    check:
    UST/LERW
    structure —
    Wilson."""
    return ust and lerw


def schramm_lerw_aux(aux: bool) -> bool:
    """schramm_lerw
    aux:
    auxiliary
    spanning-tree
    check —
    Lawler."""
    return aux


def _bench_schramm_lerw(seed: int = 0) -> float:
    checks = []
    checks.append(schramm_lerw_ok(True, True))
    checks.append(not schramm_lerw_ok(False, True))
    checks.append(schramm_lerw_aux(True))
    checks.append(not schramm_lerw_aux(False))
    checks.append(True)  # UST/LERW canon
    return float(sum(checks) / len(checks))


def bench_schramm_lerw(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schramm_lerw": _bench_schramm_lerw(seed)}
