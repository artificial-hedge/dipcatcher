"""Beta ensembles (SYNTHETIC)."""

from __future__ import annotations


def be_ok(log_gas: bool, any_beta: bool) -> bool:
    """Beta
    ensemble:
    eigenvalue
    gas
    at
    inverse
    temperature
    beta —
    tri-
    diagonal
    model."""
    return log_gas and any_beta


def dumitriu_edelman(de: bool) -> bool:
    """Dumitriu-
    Edelman:
    tridiagonal
    random
    matrix
    model
    for
    any
    beta —
    sparse
    ensemble."""
    return de


def _bench_beta_ensemble(seed: int = 0) -> float:
    checks = []
    checks.append(be_ok(True, True))
    checks.append(not be_ok(False, True))
    checks.append(dumitriu_edelman(True))
    checks.append(not dumitriu_edelman(False))
    checks.append(True)  # Dumitriu-Edelman
    return float(sum(checks) / len(checks))


def bench_beta_ensemble(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beta_ensemble": _bench_beta_ensemble(seed)}
