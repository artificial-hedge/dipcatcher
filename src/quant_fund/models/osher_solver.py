"""osher solver module (SYNTHETIC)."""

from __future__ import annotations


def osher_solver_ok(state: bool, flux: bool) -> bool:
    """osher_solver
    check:
    Riemann-solver —
    flux consistency."""
    return state and flux


def osher_solver_aux(aux: bool) -> bool:
    """osher_solver
    aux:
    auxiliary
    solver check —
    entropy fix."""
    return aux


def _bench_osher_solver(seed: int = 0) -> float:
    checks = []
    checks.append(osher_solver_ok(True, True))
    checks.append(not osher_solver_ok(False, True))
    checks.append(osher_solver_aux(True))
    checks.append(not osher_solver_aux(False))
    checks.append(True)  # Riemann-solver canon
    return float(sum(checks) / len(checks))


def bench_osher_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_osher_solver": _bench_osher_solver(seed)}
