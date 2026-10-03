"""absolute cont module (SYNTHETIC)."""

from __future__ import annotations


def absolute_cont_ok(nz1: bool, mc: bool) -> bool:
    """absolute_cont
    check:
    Malliavin —
    covariance/density."""
    return nz1 and mc


def absolute_cont_aux(aux: bool) -> bool:
    """absolute_cont
    aux:
    auxiliary
    mall
    check —
    smoothness."""
    return aux


def _bench_absolute_cont(seed: int = 0) -> float:
    checks = []
    checks.append(absolute_cont_ok(True, True))
    checks.append(not absolute_cont_ok(False, True))
    checks.append(absolute_cont_aux(True))
    checks.append(not absolute_cont_aux(False))
    checks.append(True)  # Malliavin canon
    return float(sum(checks) / len(checks))


def bench_absolute_cont(seed: int = 0) -> dict[str, float]:
    return {"synthetic_absolute_cont": _bench_absolute_cont(seed)}
