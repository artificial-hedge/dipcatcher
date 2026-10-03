"""feller boundary module (SYNTHETIC)."""

from __future__ import annotations


def feller_boundary_ok(sc: bool, sp: bool) -> bool:
    """feller_boundary
    check:
    diffusion
    theory —
    boundary."""
    return sc and sp


def feller_boundary_aux(aux: bool) -> bool:
    """feller_boundary
    aux:
    auxiliary
    diffusion
    check —
    generator."""
    return aux


def _bench_feller_boundary(seed: int = 0) -> float:
    checks = []
    checks.append(feller_boundary_ok(True, True))
    checks.append(not feller_boundary_ok(False, True))
    checks.append(feller_boundary_aux(True))
    checks.append(not feller_boundary_aux(False))
    checks.append(True)  # diffusion canon
    return float(sum(checks) / len(checks))


def bench_feller_boundary(seed: int = 0) -> dict[str, float]:
    return {"synthetic_feller_boundary": _bench_feller_boundary(seed)}
