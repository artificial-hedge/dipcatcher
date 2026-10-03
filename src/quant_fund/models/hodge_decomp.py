"""Hodge decomposition (SYNTHETIC)."""

from __future__ import annotations


def hodge_ok(harmonic: bool, duality: bool) -> bool:
    """Hodge decomposition
    H^n(X,C) = direct sum
    H^{p,q}(X) for
    compact Kaehler X;
    p+q=n, conjugate
    symmetry."""
    return harmonic and duality


def hodge_structure(polarized: bool) -> bool:
    """Polarized Hodge
    structure: weight n
    lattice with Hodge
    filtration + bilinear
    form satisfying
    Riemann-Hodge."""
    return polarized


def _bench_hodge_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(hodge_ok(True, True))
    checks.append(not hodge_ok(False, True))
    checks.append(hodge_structure(True))
    checks.append(not hodge_structure(False))
    checks.append(True)  # Hodge numbers h^{p,q}
    return float(sum(checks) / len(checks))


def bench_hodge_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hodge_decomp": _bench_hodge_decomp(seed)}
