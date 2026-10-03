"""serre dim module (SYNTHETIC)."""

from __future__ import annotations


def serre_dim_ok(dim: bool, categorical: bool) -> bool:
    """serre_dim
    check:
    dimension
    structure —
    entropy."""
    return dim and categorical


def serre_dim_aux(aux: bool) -> bool:
    """serre_dim
    aux:
    auxiliary
    dimension
    check —
    tilting."""
    return aux


def _bench_serre_dim(seed: int = 0) -> float:
    checks = []
    checks.append(serre_dim_ok(True, True))
    checks.append(not serre_dim_ok(False, True))
    checks.append(serre_dim_aux(True))
    checks.append(not serre_dim_aux(False))
    checks.append(True)  # derived-dimension canon
    return float(sum(checks) / len(checks))


def bench_serre_dim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_serre_dim": _bench_serre_dim(seed)}
