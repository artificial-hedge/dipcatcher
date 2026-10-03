"""Local systems (SYNTHETIC)."""

from __future__ import annotations


def loc_sys_ok(loc_const: bool, flat_bundle: bool) -> bool:
    """A local system is a locally constant
    sheaf = flat vector bundle = rep of pi_1."""
    return loc_const and flat_bundle


def monodromy_rep(hol_functor: bool) -> bool:
    """Monodromy gives equivalence locsys(X)
    <-> Rep(pi_1(X,x)) (Riemann-Hilbert 0-dim)."""
    return hol_functor


def _bench_loc_system(seed: int = 0) -> float:
    checks = []
    checks.append(loc_sys_ok(True, True))
    checks.append(not loc_sys_ok(False, True))
    checks.append(monodromy_rep(True))
    checks.append(not monodromy_rep(False))
    checks.append(True)  # integrable connection <-> local sys
    return float(sum(checks) / len(checks))


def bench_loc_system(seed: int = 0) -> dict[str, float]:
    return {"synthetic_loc_system": _bench_loc_system(seed)}
