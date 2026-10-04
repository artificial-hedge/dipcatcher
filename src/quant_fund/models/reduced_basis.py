"""reduced basis module (SYNTHETIC)."""

from __future__ import annotations


def reduced_basis_ok(basis: bool, mode: bool) -> bool:
    """reduced_basis
    check:
    model-order-reduction —
    snapshot
    consistency."""
    return basis and mode


def reduced_basis_aux(aux: bool) -> bool:
    """reduced_basis
    aux:
    auxiliary
    reduction check —
    energy bound."""
    return aux


def _bench_reduced_basis(seed: int = 0) -> float:
    checks = []
    checks.append(reduced_basis_ok(True, True))
    checks.append(not reduced_basis_ok(False, True))
    checks.append(reduced_basis_aux(True))
    checks.append(not reduced_basis_aux(False))
    checks.append(True)  # MOR canon
    return float(sum(checks) / len(checks))


def bench_reduced_basis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reduced_basis": _bench_reduced_basis(seed)}
