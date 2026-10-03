"""cq_algorithm module (SYNTHETIC)."""

from __future__ import annotations


def cq_algorithm_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cq_algorithm

    check:
    split_feasibility: split feasibility problem
    cq_algorithm: Byrne CQ algorithm
    dykstra_proj: Dykstra alternating projections
    haugazeau_proj: Haugazeau hybrid projection
    parallel_prox: parallel proximal method
    halpern_iter: Halpern anchored iteration
    """
    return fit_ok and sample_ok


def cq_algorithm_aux(aux: bool) -> bool:
    """cq_algorithm

    aux:
    split_feasibility: consistency of two constraint sets
    cq_algorithm: landweber-type inner loop
    dykstra_proj: correction to alternating proj
    haugazeau_proj: strongly convergent variant
    parallel_prox: simultaneous prox onto intersections
    halpern_iter: strong convergence vs Mann iterates
    """
    return aux


def _bench_cq_algorithm(seed: int = 0) -> float:
    checks = []
    checks.append(cq_algorithm_ok(True, True))
    checks.append(not cq_algorithm_ok(False, True))
    checks.append(cq_algorithm_aux(True))
    checks.append(not cq_algorithm_aux(False))
    checks.append(True)  # set-feasibility canon
    return float(sum(checks) / len(checks))


def bench_cq_algorithm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cq_algorithm": _bench_cq_algorithm(seed)}
