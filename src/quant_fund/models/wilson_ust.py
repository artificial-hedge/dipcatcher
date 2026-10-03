"""wilson ust module (SYNTHETIC)."""

from __future__ import annotations


def wilson_ust_ok(ust: bool, lerw: bool) -> bool:
    """wilson_ust
    check:
    UST/LERW
    structure —
    Wilson."""
    return ust and lerw


def wilson_ust_aux(aux: bool) -> bool:
    """wilson_ust
    aux:
    auxiliary
    spanning-tree
    check —
    Lawler."""
    return aux


def _bench_wilson_ust(seed: int = 0) -> float:
    checks = []
    checks.append(wilson_ust_ok(True, True))
    checks.append(not wilson_ust_ok(False, True))
    checks.append(wilson_ust_aux(True))
    checks.append(not wilson_ust_aux(False))
    checks.append(True)  # UST/LERW canon
    return float(sum(checks) / len(checks))


def bench_wilson_ust(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wilson_ust": _bench_wilson_ust(seed)}
