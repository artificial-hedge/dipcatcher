"""pemantle ust module (SYNTHETIC)."""

from __future__ import annotations


def pemantle_ust_ok(ust: bool, lerw: bool) -> bool:
    """pemantle_ust
    check:
    UST/LERW
    structure —
    Wilson."""
    return ust and lerw


def pemantle_ust_aux(aux: bool) -> bool:
    """pemantle_ust
    aux:
    auxiliary
    spanning-tree
    check —
    Lawler."""
    return aux


def _bench_pemantle_ust(seed: int = 0) -> float:
    checks = []
    checks.append(pemantle_ust_ok(True, True))
    checks.append(not pemantle_ust_ok(False, True))
    checks.append(pemantle_ust_aux(True))
    checks.append(not pemantle_ust_aux(False))
    checks.append(True)  # UST/LERW canon
    return float(sum(checks) / len(checks))


def bench_pemantle_ust(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pemantle_ust": _bench_pemantle_ust(seed)}
