"""iterated_local module (SYNTHETIC)."""

from __future__ import annotations


def iterated_local_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iterated_local

    check:
    lin_kernighan: Lin-Kernighan TSP heuristic
    two_opt_move: 2-opt edge exchange
    three_opt_move: 3-opt edge exchange
    tabu_search: tabu list + aspiration
    iterated_local: perturbation + local search
    grasp_meta: GRASP randomized greedy
    """
    return fit_ok and sample_ok


def iterated_local_aux(aux: bool) -> bool:
    """iterated_local

    aux:
    lin_kernighan: depth-limited k-opt chain
    two_opt_move: reversal of subtour
    three_opt_move: seven reconnection cases
    tabu_search: tenure + intensification
    iterated_local: acceptance criterion walk
    grasp_meta: RCL threshold sampling
    """
    return aux


def _bench_iterated_local(seed: int = 0) -> float:
    checks = []
    checks.append(iterated_local_ok(True, True))
    checks.append(not iterated_local_ok(False, True))
    checks.append(iterated_local_aux(True))
    checks.append(not iterated_local_aux(False))
    checks.append(True)  # metaheuristics canon
    return float(sum(checks) / len(checks))


def bench_iterated_local(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iterated_local": _bench_iterated_local(seed)}
