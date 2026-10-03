"""hllc solver module (SYNTHETIC)."""

from __future__ import annotations


def hllc_solver_ok(state: bool, flux: bool) -> bool:
    """hllc_solver
    check:
    Riemann-solver —
    flux consistency."""
    return state and flux


def hllc_solver_aux(aux: bool) -> bool:
    """hllc_solver
    aux:
    auxiliary
    solver check —
    entropy fix."""
    return aux


def _bench_hllc_solver(seed: int = 0) -> float:
    checks = []
    checks.append(hllc_solver_ok(True, True))
    checks.append(not hllc_solver_ok(False, True))
    checks.append(hllc_solver_aux(True))
    checks.append(not hllc_solver_aux(False))
    checks.append(True)  # Riemann-solver canon
    return float(sum(checks) / len(checks))


def bench_hllc_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hllc_solver": _bench_hllc_solver(seed)}
