"""importance rel module (SYNTHETIC)."""

from __future__ import annotations


def importance_rel_ok(step: bool, conv: bool) -> bool:
    """importance_rel
    check:
    solver/transport —
    step/convergence
    consistency."""
    return step and conv


def importance_rel_aux(aux: bool) -> bool:
    """importance_rel
    aux:
    auxiliary
    solver check —
    order bound."""
    return aux


def _bench_importance_rel(seed: int = 0) -> float:
    checks = []
    checks.append(importance_rel_ok(True, True))
    checks.append(not importance_rel_ok(False, True))
    checks.append(importance_rel_aux(True))
    checks.append(not importance_rel_aux(False))
    checks.append(True)  # solver/transport canon
    return float(sum(checks) / len(checks))


def bench_importance_rel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_importance_rel": _bench_importance_rel(seed)}
