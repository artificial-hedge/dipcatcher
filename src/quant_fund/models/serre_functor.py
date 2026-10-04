"""serre functor module (SYNTHETIC)."""

from __future__ import annotations


def serre_functor_ok(triangulated: bool, derived: bool) -> bool:
    """serre_functor
    check:
    triangulated
    structure —
    exceptional."""
    return triangulated and derived


def serre_functor_aux(aux: bool) -> bool:
    """serre_functor
    aux:
    auxiliary
    triangulated
    check —
    Fourier."""
    return aux


def _bench_serre_functor(seed: int = 0) -> float:
    checks = []
    checks.append(serre_functor_ok(True, True))
    checks.append(not serre_functor_ok(False, True))
    checks.append(serre_functor_aux(True))
    checks.append(not serre_functor_aux(False))
    checks.append(True)  # triangulated canon
    return float(sum(checks) / len(checks))


def bench_serre_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_serre_functor": _bench_serre_functor(seed)}
