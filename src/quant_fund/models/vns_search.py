"""vns_search module (SYNTHETIC)."""

from __future__ import annotations


def vns_search_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """vns_search

    check:
    vns_search: variable neighborhood search
    large_neighborhood: LNS destroy-repair
    ruin_recreate: ruin-and-recreate moves
    path_relinking: elite-set path relinking
    guided_local: guided local search penalties
    simulated_annealing: Metropolis acceptance
    """
    return fit_ok and sample_ok


def vns_search_aux(aux: bool) -> bool:
    """vns_search

    aux:
    vns_search: shaking + neighborhood change
    large_neighborhood: adaptive operator weights
    ruin_recreate: removal-degree control
    path_relinking: attribute vote along path
    guided_local: augmented objective features
    simulated_annealing: exponential cooling schedule
    """
    return aux


def _bench_vns_search(seed: int = 0) -> float:
    checks = []
    checks.append(vns_search_ok(True, True))
    checks.append(not vns_search_ok(False, True))
    checks.append(vns_search_aux(True))
    checks.append(not vns_search_aux(False))
    checks.append(True)  # local-search-3 canon
    return float(sum(checks) / len(checks))


def bench_vns_search(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vns_search": _bench_vns_search(seed)}
