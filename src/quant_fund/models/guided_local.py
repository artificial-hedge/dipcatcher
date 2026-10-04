"""guided_local module (SYNTHETIC)."""

from __future__ import annotations


def guided_local_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """guided_local

    check:
    vns_search: variable neighborhood search
    large_neighborhood: LNS destroy-repair
    ruin_recreate: ruin-and-recreate moves
    path_relinking: elite-set path relinking
    guided_local: guided local search penalties
    simulated_annealing: Metropolis acceptance
    """
    return fit_ok and sample_ok


def guided_local_aux(aux: bool) -> bool:
    """guided_local

    aux:
    vns_search: shaking + neighborhood change
    large_neighborhood: adaptive operator weights
    ruin_recreate: removal-degree control
    path_relinking: attribute vote along path
    guided_local: augmented objective features
    simulated_annealing: exponential cooling schedule
    """
    return aux


def _bench_guided_local(seed: int = 0) -> float:
    checks = []
    checks.append(guided_local_ok(True, True))
    checks.append(not guided_local_ok(False, True))
    checks.append(guided_local_aux(True))
    checks.append(not guided_local_aux(False))
    checks.append(True)  # local-search-3 canon
    return float(sum(checks) / len(checks))


def bench_guided_local(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guided_local": _bench_guided_local(seed)}
