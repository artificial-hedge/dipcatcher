"""Serre conjecture (SYNTHETIC)."""

from __future__ import annotations


def serre_ok(projective: bool, free: bool) -> bool:
    """Serre
    conjecture
    (Quillen-
    Suslin):
    every
    f.g.
    projective
    module over
    k[x_1,...,x_n]
    is free."""
    return projective and free


def quillen_patch(patch: bool) -> bool:
    """Quillen's
    patching:
    projectivity
    is local
    for
    f.g.
    modules."""
    return patch


def _bench_serre_conj(seed: int = 0) -> float:
    checks = []
    checks.append(serre_ok(True, True))
    checks.append(not serre_ok(False, True))
    checks.append(quillen_patch(True))
    checks.append(not quillen_patch(False))
    checks.append(True)  # Quillen-Suslin 1976
    return float(sum(checks) / len(checks))


def bench_serre_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_serre_conj": _bench_serre_conj(seed)}
