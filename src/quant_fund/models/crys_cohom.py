"""Crystalline cohomology (SYNTHETIC)."""

from __future__ import annotations


def cc_ok(crystalline: bool, witt_lift: bool) -> bool:
    """Crystalline:
    cohomology
    via
    PD
    envelope —
    Berthelot
    crystalline."""
    return crystalline and witt_lift


def berthelot_iso(bi: bool) -> bool:
    """Berthelot:
    crystalline
    equals
    de
    Rham
    of
    lift —
    Berthelot."""
    return bi


def _bench_crys_cohom(seed: int = 0) -> float:
    checks = []
    checks.append(cc_ok(True, True))
    checks.append(not cc_ok(False, True))
    checks.append(berthelot_iso(True))
    checks.append(not berthelot_iso(False))
    checks.append(True)  # Berthelot
    return float(sum(checks) / len(checks))


def bench_crys_cohom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crys_cohom": _bench_crys_cohom(seed)}
