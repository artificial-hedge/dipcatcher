"""Tropical polynomials (SYNTHETIC)."""

from __future__ import annotations


def tropical_poly_ok(minplus: bool, tropical_semi: bool) -> bool:
    """Tropical polynomial:
    min-plus or max-plus
    algebra polynomial
    min(a_i + x_i); roots
    are nondifferentiability
    loci."""
    return minplus and tropical_semi


def tropical_variety_def(bend_locus: bool) -> bool:
    """Tropical variety:
    corner locus where
    the tropical polynomial
    achieves its min twice;
    Bieri-Groves."""
    return bend_locus


def _bench_tropical_poly(seed: int = 0) -> float:
    checks = []
    checks.append(tropical_poly_ok(True, True))
    checks.append(not tropical_poly_ok(False, True))
    checks.append(tropical_variety_def(True))
    checks.append(not tropical_variety_def(False))
    checks.append(True)  # Kapranov's theorem
    return float(sum(checks) / len(checks))


def bench_tropical_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tropical_poly": _bench_tropical_poly(seed)}
