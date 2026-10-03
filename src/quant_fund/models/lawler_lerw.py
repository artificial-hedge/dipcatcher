"""lawler lerw module (SYNTHETIC)."""

from __future__ import annotations


def lawler_lerw_ok(ust: bool, lerw: bool) -> bool:
    """lawler_lerw
    check:
    UST/LERW
    structure —
    Wilson."""
    return ust and lerw


def lawler_lerw_aux(aux: bool) -> bool:
    """lawler_lerw
    aux:
    auxiliary
    spanning-tree
    check —
    Lawler."""
    return aux


def _bench_lawler_lerw(seed: int = 0) -> float:
    checks = []
    checks.append(lawler_lerw_ok(True, True))
    checks.append(not lawler_lerw_ok(False, True))
    checks.append(lawler_lerw_aux(True))
    checks.append(not lawler_lerw_aux(False))
    checks.append(True)  # UST/LERW canon
    return float(sum(checks) / len(checks))


def bench_lawler_lerw(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lawler_lerw": _bench_lawler_lerw(seed)}
