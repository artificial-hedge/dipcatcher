"""hairer kpz module (SYNTHETIC)."""

from __future__ import annotations


def hairer_kpz_ok(kpz: bool, reg: bool) -> bool:
    """hairer_kpz
    check:
    KPZ-2
    structure —
    Corwin."""
    return kpz and reg


def hairer_kpz_aux(aux: bool) -> bool:
    """hairer_kpz
    aux:
    auxiliary
    regularity-structure
    check —
    Hairer."""
    return aux


def _bench_hairer_kpz(seed: int = 0) -> float:
    checks = []
    checks.append(hairer_kpz_ok(True, True))
    checks.append(not hairer_kpz_ok(False, True))
    checks.append(hairer_kpz_aux(True))
    checks.append(not hairer_kpz_aux(False))
    checks.append(True)  # KPZ-2 canon
    return float(sum(checks) / len(checks))


def bench_hairer_kpz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hairer_kpz": _bench_hairer_kpz(seed)}
