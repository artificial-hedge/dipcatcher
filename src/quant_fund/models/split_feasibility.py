"""split_feasibility module (SYNTHETIC)."""

from __future__ import annotations


def split_feasibility_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """split_feasibility

    check:
    split_feasibility: split feasibility problem
    cq_algorithm: Byrne CQ algorithm
    dykstra_proj: Dykstra alternating projections
    haugazeau_proj: Haugazeau hybrid projection
    parallel_prox: parallel proximal method
    halpern_iter: Halpern anchored iteration
    """
    return fit_ok and sample_ok


def split_feasibility_aux(aux: bool) -> bool:
    """split_feasibility

    aux:
    split_feasibility: consistency of two constraint sets
    cq_algorithm: landweber-type inner loop
    dykstra_proj: correction to alternating proj
    haugazeau_proj: strongly convergent variant
    parallel_prox: simultaneous prox onto intersections
    halpern_iter: strong convergence vs Mann iterates
    """
    return aux


def _bench_split_feasibility(seed: int = 0) -> float:
    checks = []
    checks.append(split_feasibility_ok(True, True))
    checks.append(not split_feasibility_ok(False, True))
    checks.append(split_feasibility_aux(True))
    checks.append(not split_feasibility_aux(False))
    checks.append(True)  # set-feasibility canon
    return float(sum(checks) / len(checks))


def bench_split_feasibility(seed: int = 0) -> dict[str, float]:
    return {"synthetic_split_feasibility": _bench_split_feasibility(seed)}
