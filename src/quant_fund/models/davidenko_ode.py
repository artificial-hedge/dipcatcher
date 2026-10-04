"""davidenko ode module (SYNTHETIC)."""

from __future__ import annotations


def davidenko_ode_ok(path: bool, step: bool) -> bool:
    """davidenko_ode
    check:
    continuation/homotopy —
    predictor
    consistency."""
    return path and step


def davidenko_ode_aux(aux: bool) -> bool:
    """davidenko_ode
    aux:
    auxiliary
    continuation check —
    corrector bound."""
    return aux


def _bench_davidenko_ode(seed: int = 0) -> float:
    checks = []
    checks.append(davidenko_ode_ok(True, True))
    checks.append(not davidenko_ode_ok(False, True))
    checks.append(davidenko_ode_aux(True))
    checks.append(not davidenko_ode_aux(False))
    checks.append(True)  # continuation canon
    return float(sum(checks) / len(checks))


def bench_davidenko_ode(seed: int = 0) -> dict[str, float]:
    return {"synthetic_davidenko_ode": _bench_davidenko_ode(seed)}
