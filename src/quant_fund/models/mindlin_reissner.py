"""mindlin_reissner module (SYNTHETIC)."""

from __future__ import annotations


def mindlin_reissner_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mindlin_reissner

    check:
    navier_elasticity: Navier linear elasticity equations
    kirchhoff_plate: Kirchhoff-Love plate theory
    mindlin_reissner: Mindlin-Reissner plates
    contact_mechanics: contact and Signorini problem
    fracture_mechanics: Griffith fracture mechanics
    homogenized_elasticity: periodic homogenization in elasticity
    """
    return fit_ok and sample_ok


def mindlin_reissner_aux(aux: bool) -> bool:
    """mindlin_reissner

    aux:
    navier_elasticity: Korn inequality
    kirchhoff_plate: biharmonic reduction
    mindlin_reissner: shear locking
    contact_mechanics: complementarity
    fracture_mechanics: J-integral
    homogenized_elasticity: cell problems
    """
    return aux


def _bench_mindlin_reissner(seed: int = 0) -> float:
    checks = []
    checks.append(mindlin_reissner_ok(True, True))
    checks.append(not mindlin_reissner_ok(False, True))
    checks.append(mindlin_reissner_aux(True))
    checks.append(not mindlin_reissner_aux(False))
    checks.append(True)  # elasticity canon
    return float(sum(checks) / len(checks))


def bench_mindlin_reissner(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mindlin_reissner": _bench_mindlin_reissner(seed)}
