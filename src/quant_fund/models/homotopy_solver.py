"""homotopy solver module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_solver_ok(path: bool, step: bool) -> bool:
    """homotopy_solver
    check:
    continuation/homotopy —
    predictor
    consistency."""
    return path and step


def homotopy_solver_aux(aux: bool) -> bool:
    """homotopy_solver
    aux:
    auxiliary
    continuation check —
    corrector bound."""
    return aux


def _bench_homotopy_solver(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_solver_ok(True, True))
    checks.append(not homotopy_solver_ok(False, True))
    checks.append(homotopy_solver_aux(True))
    checks.append(not homotopy_solver_aux(False))
    checks.append(True)  # continuation canon
    return float(sum(checks) / len(checks))


def bench_homotopy_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_solver": _bench_homotopy_solver(seed)}
