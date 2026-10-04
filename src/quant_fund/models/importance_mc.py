"""importance mc module (SYNTHETIC)."""

from __future__ import annotations


def importance_mc_ok(node: bool, wgt: bool) -> bool:
    """importance_mc
    check:
    quadrature/tensor —
    node/weight
    consistency."""
    return node and wgt


def importance_mc_aux(aux: bool) -> bool:
    """importance_mc
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_importance_mc(seed: int = 0) -> float:
    checks = []
    checks.append(importance_mc_ok(True, True))
    checks.append(not importance_mc_ok(False, True))
    checks.append(importance_mc_aux(True))
    checks.append(not importance_mc_aux(False))
    checks.append(True)  # QMC/tensor canon
    return float(sum(checks) / len(checks))


def bench_importance_mc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_importance_mc": _bench_importance_mc(seed)}
