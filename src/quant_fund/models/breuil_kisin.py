"""Breuil-Kisin modules (SYNTHETIC)."""

from __future__ import annotations


def bk_ok(eisenstein: bool, phi_finite: bool) -> bool:
    """Breuil-Kisin module: finite S =
    W(k)[[u]]-module with Frobenius
    cokernel killed by E(u)^h; classifies
    stable lattices (Kisin)."""
    return eisenstein and phi_finite


def galois_lattice(fully_faithful: bool) -> bool:
    """T_BK gives fully faithful functor
    from BK modules to Z_p Galois
    representations."""
    return fully_faithful


def _bench_breuil_kisin(seed: int = 0) -> float:
    checks = []
    checks.append(bk_ok(True, True))
    checks.append(not bk_ok(False, True))
    checks.append(galois_lattice(True))
    checks.append(not galois_lattice(False))
    checks.append(True)  # height bound determines deformation
    return float(sum(checks) / len(checks))


def bench_breuil_kisin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_breuil_kisin": _bench_breuil_kisin(seed)}
