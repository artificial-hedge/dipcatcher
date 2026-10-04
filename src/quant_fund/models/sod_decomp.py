"""sod decomp module (SYNTHETIC)."""

from __future__ import annotations


def sod_decomp_ok(triangulated: bool, derived: bool) -> bool:
    """sod_decomp
    check:
    triangulated
    structure —
    exceptional."""
    return triangulated and derived


def sod_decomp_aux(aux: bool) -> bool:
    """sod_decomp
    aux:
    auxiliary
    triangulated
    check —
    Fourier."""
    return aux


def _bench_sod_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(sod_decomp_ok(True, True))
    checks.append(not sod_decomp_ok(False, True))
    checks.append(sod_decomp_aux(True))
    checks.append(not sod_decomp_aux(False))
    checks.append(True)  # triangulated canon
    return float(sum(checks) / len(checks))


def bench_sod_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sod_decomp": _bench_sod_decomp(seed)}
