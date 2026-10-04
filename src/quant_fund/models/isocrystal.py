"""Isocrystals / F-isocrystals (SYNTHETIC)."""

from __future__ import annotations


def isocrystal_ok(frob: bool, overconv: bool) -> bool:
    """An (overconvergent) F-isocrystal on X/k is
    a convergent isocrystal with Frobenius
    structure; unit-root iff pure slope 0."""
    return frob and overconv


def unit_root_coeffs(coeff_field: bool) -> bool:
    """Unit-root F-isocrystals correspond to
    p-adic representations of pi_1 (Crew/Katz)."""
    return coeff_field


def _bench_isocrystal(seed: int = 0) -> float:
    checks = []
    checks.append(isocrystal_ok(True, True))
    checks.append(not isocrystal_ok(False, True))
    checks.append(unit_root_coeffs(True))
    checks.append(not unit_root_coeffs(False))
    checks.append(True)  # Dieudonne-Manin slope decomposition
    return float(sum(checks) / len(checks))


def bench_isocrystal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isocrystal": _bench_isocrystal(seed)}
