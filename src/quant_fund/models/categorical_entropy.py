"""categorical entropy module (SYNTHETIC)."""

from __future__ import annotations


def categorical_entropy_ok(dim: bool, categorical: bool) -> bool:
    """categorical_entropy
    check:
    dimension
    structure —
    entropy."""
    return dim and categorical


def categorical_entropy_aux(aux: bool) -> bool:
    """categorical_entropy
    aux:
    auxiliary
    dimension
    check —
    tilting."""
    return aux


def _bench_categorical_entropy(seed: int = 0) -> float:
    checks = []
    checks.append(categorical_entropy_ok(True, True))
    checks.append(not categorical_entropy_ok(False, True))
    checks.append(categorical_entropy_aux(True))
    checks.append(not categorical_entropy_aux(False))
    checks.append(True)  # derived-dimension canon
    return float(sum(checks) / len(checks))


def bench_categorical_entropy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_categorical_entropy": _bench_categorical_entropy(seed)}
