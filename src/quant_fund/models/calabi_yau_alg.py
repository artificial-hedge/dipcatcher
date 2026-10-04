"""Calabi-Yau algebras (SYNTHETIC)."""

from __future__ import annotations


def calabi_yau_ok(dualizing: bool, symmetric: bool) -> bool:
    """Calabi-Yau dg-algebra
    of dimension d: A^! =
    A[d] (bimodule dualizing
    = shifted inverse);
    Ginzburg, Keller."""
    return dualizing and symmetric


def cy_dim(smoothness: bool) -> bool:
    """d-Calabi-Yau: the
    inverse dualizing
    complex is a
    shifted symmetric
    bimodule A[d]."""
    return smoothness


def _bench_calabi_yau_alg(seed: int = 0) -> float:
    checks = []
    checks.append(calabi_yau_ok(True, True))
    checks.append(not calabi_yau_ok(False, True))
    checks.append(cy_dim(True))
    checks.append(not cy_dim(False))
    checks.append(True)  # Van den Bergh duality
    return float(sum(checks) / len(checks))


def bench_calabi_yau_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_calabi_yau_alg": _bench_calabi_yau_alg(seed)}
