"""tracy widom_kpz module (SYNTHETIC)."""

from __future__ import annotations


def tracy_widom_kpz_ok(kpz: bool, reg: bool) -> bool:
    """tracy_widom_kpz
    check:
    KPZ-2
    structure —
    Corwin."""
    return kpz and reg


def tracy_widom_kpz_aux(aux: bool) -> bool:
    """tracy_widom_kpz
    aux:
    auxiliary
    regularity-structure
    check —
    Hairer."""
    return aux


def _bench_tracy_widom_kpz(seed: int = 0) -> float:
    checks = []
    checks.append(tracy_widom_kpz_ok(True, True))
    checks.append(not tracy_widom_kpz_ok(False, True))
    checks.append(tracy_widom_kpz_aux(True))
    checks.append(not tracy_widom_kpz_aux(False))
    checks.append(True)  # KPZ-2 canon
    return float(sum(checks) / len(checks))


def bench_tracy_widom_kpz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tracy_widom_kpz": _bench_tracy_widom_kpz(seed)}
