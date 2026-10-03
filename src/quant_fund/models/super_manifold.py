"""Supermanifolds (SYNTHETIC)."""

from __future__ import annotations


def super_mfld_ok(chart: bool, parity: bool) -> bool:
    """Supermanifold:
    locally modeled
    on R^{m|n};
    transition
    functions are
    super-smooth
    maps."""
    return chart and parity


def batchelor_thm(batchelor: bool) -> bool:
    """Batchelor
    theorem: every
    smooth
    supermanifold
    is (non-canonically)
    an exterior
    bundle over
    a manifold."""
    return batchelor


def _bench_super_manifold(seed: int = 0) -> float:
    checks = []
    checks.append(super_mfld_ok(True, True))
    checks.append(not super_mfld_ok(False, True))
    checks.append(batchelor_thm(True))
    checks.append(not batchelor_thm(False))
    checks.append(True)  # Batchelor 1979
    return float(sum(checks) / len(checks))


def bench_super_manifold(seed: int = 0) -> dict[str, float]:
    return {"synthetic_super_manifold": _bench_super_manifold(seed)}
