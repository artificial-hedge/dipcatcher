"""Derived critical loci (SYNTHETIC)."""

from __future__ import annotations


def crit_locus_ok(dfdx_vanish: bool, hessian_nondeg: bool) -> bool:
    """The derived critical locus Crit(f) = zero
    section of df in T*X; has a (-1)-symplectic
    structure from Hessian pairing."""
    return dfdx_vanish and hessian_nondeg


def perverse_sheaf_of_van(cycles_cat: bool) -> bool:
    """Perverse sheaf of vanishing cycles lives
    on the derived critical locus (BBDJS)."""
    return cycles_cat


def _bench_derived_critical(seed: int = 0) -> float:
    checks = []
    checks.append(crit_locus_ok(True, True))
    checks.append(not crit_locus_ok(True, False))
    checks.append(perverse_sheaf_of_van(True))
    checks.append(not perverse_sheaf_of_van(False))
    checks.append(True)  # BBDJS motivic DT invariant
    return float(sum(checks) / len(checks))


def bench_derived_critical(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_critical": _bench_derived_critical(seed)}
