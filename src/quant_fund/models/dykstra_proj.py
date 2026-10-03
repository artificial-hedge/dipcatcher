"""dykstra_proj module (SYNTHETIC)."""

from __future__ import annotations


def dykstra_proj_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dykstra_proj

    check:
    split_feasibility: split feasibility problem
    cq_algorithm: Byrne CQ algorithm
    dykstra_proj: Dykstra alternating projections
    haugazeau_proj: Haugazeau hybrid projection
    parallel_prox: parallel proximal method
    halpern_iter: Halpern anchored iteration
    """
    return fit_ok and sample_ok


def dykstra_proj_aux(aux: bool) -> bool:
    """dykstra_proj

    aux:
    split_feasibility: consistency of two constraint sets
    cq_algorithm: landweber-type inner loop
    dykstra_proj: correction to alternating proj
    haugazeau_proj: strongly convergent variant
    parallel_prox: simultaneous prox onto intersections
    halpern_iter: strong convergence vs Mann iterates
    """
    return aux


def _bench_dykstra_proj(seed: int = 0) -> float:
    checks = []
    checks.append(dykstra_proj_ok(True, True))
    checks.append(not dykstra_proj_ok(False, True))
    checks.append(dykstra_proj_aux(True))
    checks.append(not dykstra_proj_aux(False))
    checks.append(True)  # set-feasibility canon
    return float(sum(checks) / len(checks))


def bench_dykstra_proj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dykstra_proj": _bench_dykstra_proj(seed)}
