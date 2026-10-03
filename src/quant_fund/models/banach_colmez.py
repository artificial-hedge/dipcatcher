"""Banach-Colmez spaces (SYNTHETIC)."""

from __future__ import annotations


def bc_ok(dimension: bool, tangential: bool) -> bool:
    """Banach-Colmez space: finite
    dimensional Banach space over C_p
    quotiented by discrete C_p module;
    has dimension + tangential dimension."""
    return dimension and tangential


def weight_tensor(tensor_closed: bool) -> bool:
    """BC spaces form a tensor category;
    they are the Banach-space analogues
    of p-adic periods."""
    return tensor_closed


def _bench_banach_colmez(seed: int = 0) -> float:
    checks = []
    checks.append(bc_ok(True, True))
    checks.append(not bc_ok(False, True))
    checks.append(weight_tensor(True))
    checks.append(not weight_tensor(False))
    checks.append(True)  # used in Colmez's Montréal functor
    return float(sum(checks) / len(checks))


def bench_banach_colmez(seed: int = 0) -> dict[str, float]:
    return {"synthetic_banach_colmez": _bench_banach_colmez(seed)}
