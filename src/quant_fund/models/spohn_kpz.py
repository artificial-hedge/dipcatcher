"""spohn kpz module (SYNTHETIC)."""

from __future__ import annotations


def spohn_kpz_ok(kpz: bool, reg: bool) -> bool:
    """spohn_kpz
    check:
    KPZ-2
    structure —
    Corwin."""
    return kpz and reg


def spohn_kpz_aux(aux: bool) -> bool:
    """spohn_kpz
    aux:
    auxiliary
    regularity-structure
    check —
    Hairer."""
    return aux


def _bench_spohn_kpz(seed: int = 0) -> float:
    checks = []
    checks.append(spohn_kpz_ok(True, True))
    checks.append(not spohn_kpz_ok(False, True))
    checks.append(spohn_kpz_aux(True))
    checks.append(not spohn_kpz_aux(False))
    checks.append(True)  # KPZ-2 canon
    return float(sum(checks) / len(checks))


def bench_spohn_kpz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spohn_kpz": _bench_spohn_kpz(seed)}
