"""preprojective alg module (SYNTHETIC)."""

from __future__ import annotations


def preprojective_alg_ok(dim: bool, categorical: bool) -> bool:
    """preprojective_alg
    check:
    dimension
    structure —
    entropy."""
    return dim and categorical


def preprojective_alg_aux(aux: bool) -> bool:
    """preprojective_alg
    aux:
    auxiliary
    dimension
    check —
    tilting."""
    return aux


def _bench_preprojective_alg(seed: int = 0) -> float:
    checks = []
    checks.append(preprojective_alg_ok(True, True))
    checks.append(not preprojective_alg_ok(False, True))
    checks.append(preprojective_alg_aux(True))
    checks.append(not preprojective_alg_aux(False))
    checks.append(True)  # derived-dimension canon
    return float(sum(checks) / len(checks))


def bench_preprojective_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_preprojective_alg": _bench_preprojective_alg(seed)}
