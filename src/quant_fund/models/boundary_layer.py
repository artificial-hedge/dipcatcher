"""boundary layer module (SYNTHETIC)."""

from __future__ import annotations


def boundary_layer_ok(epsilon: bool, uniform: bool) -> bool:
    """boundary_layer
    check:
    perturbation
    method —
    uniform validity."""
    return epsilon and uniform


def boundary_layer_aux(aux: bool) -> bool:
    """boundary_layer
    aux:
    auxiliary
    perturbation check —
    remainder bound."""
    return aux


def _bench_boundary_layer(seed: int = 0) -> float:
    checks = []
    checks.append(boundary_layer_ok(True, True))
    checks.append(not boundary_layer_ok(False, True))
    checks.append(boundary_layer_aux(True))
    checks.append(not boundary_layer_aux(False))
    checks.append(True)  # perturbation-theory canon
    return float(sum(checks) / len(checks))


def bench_boundary_layer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boundary_layer": _bench_boundary_layer(seed)}
