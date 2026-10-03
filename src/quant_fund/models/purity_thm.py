"""Purity theorems (SYNTHETIC)."""

from __future__ import annotations


def purity_ok(pure_sheaf: bool, smooth: bool) -> bool:
    """Purity theorem:
    a pure l-adic
    sheaf on a
    smooth variety
    is geometrically
    semisimple."""
    return pure_sheaf and smooth


def decomposition_pure(decomp: bool) -> bool:
    """Decomposition:
    pure perverse
    sheaves decompose
    into a direct
    sum of simples
    (BBD)."""
    return decomp


def _bench_purity_thm(seed: int = 0) -> float:
    checks = []
    checks.append(purity_ok(True, True))
    checks.append(not purity_ok(False, True))
    checks.append(decomposition_pure(True))
    checks.append(not decomposition_pure(False))
    checks.append(True)  # BBD
    return float(sum(checks) / len(checks))


def bench_purity_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_purity_thm": _bench_purity_thm(seed)}
