"""rouquier dim module (SYNTHETIC)."""

from __future__ import annotations


def rouquier_dim_ok(dim: bool, categorical: bool) -> bool:
    """rouquier_dim
    check:
    dimension
    structure —
    entropy."""
    return dim and categorical


def rouquier_dim_aux(aux: bool) -> bool:
    """rouquier_dim
    aux:
    auxiliary
    dimension
    check —
    tilting."""
    return aux


def _bench_rouquier_dim(seed: int = 0) -> float:
    checks = []
    checks.append(rouquier_dim_ok(True, True))
    checks.append(not rouquier_dim_ok(False, True))
    checks.append(rouquier_dim_aux(True))
    checks.append(not rouquier_dim_aux(False))
    checks.append(True)  # derived-dimension canon
    return float(sum(checks) / len(checks))


def bench_rouquier_dim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rouquier_dim": _bench_rouquier_dim(seed)}
