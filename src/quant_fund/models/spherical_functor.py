"""spherical functor module (SYNTHETIC)."""

from __future__ import annotations


def spherical_functor_ok(triangulated: bool, derived: bool) -> bool:
    """spherical_functor
    check:
    triangulated
    structure —
    exceptional."""
    return triangulated and derived


def spherical_functor_aux(aux: bool) -> bool:
    """spherical_functor
    aux:
    auxiliary
    triangulated
    check —
    Fourier."""
    return aux


def _bench_spherical_functor(seed: int = 0) -> float:
    checks = []
    checks.append(spherical_functor_ok(True, True))
    checks.append(not spherical_functor_ok(False, True))
    checks.append(spherical_functor_aux(True))
    checks.append(not spherical_functor_aux(False))
    checks.append(True)  # triangulated canon
    return float(sum(checks) / len(checks))


def bench_spherical_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spherical_functor": _bench_spherical_functor(seed)}
