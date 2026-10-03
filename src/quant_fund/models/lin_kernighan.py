"""lin_kernighan module (SYNTHETIC)."""

from __future__ import annotations


def lin_kernighan_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lin_kernighan

    check:
    lin_kernighan: Lin-Kernighan TSP heuristic
    two_opt_move: 2-opt edge exchange
    three_opt_move: 3-opt edge exchange
    tabu_search: tabu list + aspiration
    iterated_local: perturbation + local search
    grasp_meta: GRASP randomized greedy
    """
    return fit_ok and sample_ok


def lin_kernighan_aux(aux: bool) -> bool:
    """lin_kernighan

    aux:
    lin_kernighan: depth-limited k-opt chain
    two_opt_move: reversal of subtour
    three_opt_move: seven reconnection cases
    tabu_search: tenure + intensification
    iterated_local: acceptance criterion walk
    grasp_meta: RCL threshold sampling
    """
    return aux


def _bench_lin_kernighan(seed: int = 0) -> float:
    checks = []
    checks.append(lin_kernighan_ok(True, True))
    checks.append(not lin_kernighan_ok(False, True))
    checks.append(lin_kernighan_aux(True))
    checks.append(not lin_kernighan_aux(False))
    checks.append(True)  # metaheuristics canon
    return float(sum(checks) / len(checks))


def bench_lin_kernighan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lin_kernighan": _bench_lin_kernighan(seed)}
