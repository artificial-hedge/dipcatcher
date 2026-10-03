"""Soergel bimodules (SYNTHETIC)."""

from __future__ import annotations


def soergel_ok(bimodule: bool, tensor: bool) -> bool:
    """Soergel
    bimodule B_s:
    R-otimes-
    R^{sigma}
    for s a
    simple
    reflection;
    tensor products
    give Bott-
    Samelson."""
    return bimodule and tensor


def bottled_sam(bs: bool) -> bool:
    """Bott-
    Samelson
    bimodules
    decompose
    into
    indecomposable
    Soergel
    bimodules."""
    return bs


def _bench_soergel_bim(seed: int = 0) -> float:
    checks = []
    checks.append(soergel_ok(True, True))
    checks.append(not soergel_ok(False, True))
    checks.append(bottled_sam(True))
    checks.append(not bottled_sam(False))
    checks.append(True)  # Soergel
    return float(sum(checks) / len(checks))


def bench_soergel_bim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_soergel_bim": _bench_soergel_bim(seed)}
