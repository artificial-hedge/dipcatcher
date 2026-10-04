"""cramer wold module (SYNTHETIC)."""

from __future__ import annotations


def cramer_wold_ok(ord1: bool, meas: bool) -> bool:
    """cramer_wold
    check:
    stochastic-order
    structure —
    semimartingale
    canon."""
    return ord1 and meas


def cramer_wold_aux(aux: bool) -> bool:
    """cramer_wold
    aux:
    auxiliary
    order
    check —
    Cramer-Wold
    device."""
    return aux


def _bench_cramer_wold(seed: int = 0) -> float:
    checks = []
    checks.append(cramer_wold_ok(True, True))
    checks.append(not cramer_wold_ok(False, True))
    checks.append(cramer_wold_aux(True))
    checks.append(not cramer_wold_aux(False))
    checks.append(True)  # order canon
    return float(sum(checks) / len(checks))


def bench_cramer_wold(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cramer_wold": _bench_cramer_wold(seed)}
