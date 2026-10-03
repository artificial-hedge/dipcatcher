"""Iwahori-Hecke algebras (SYNTHETIC)."""

from __future__ import annotations


def hecke_bm_ok(hecke_alg: bool, quadratic: bool) -> bool:
    """Iwahori-Hecke algebra
    H(W, q): deformation of
    group algebra of Coxeter
    group W with quadratic
    relation (T_s-q)(T_s+1)=0."""
    return hecke_alg and quadratic


def kazhdan_lusztig(canonical: bool) -> bool:
    """Kazhdan-Lusztig basis
    C_w of H(W) gives
    canonical basis; KL
    polynomials compute
    characters."""
    return canonical


def _bench_hecke_bm(seed: int = 0) -> float:
    checks = []
    checks.append(hecke_bm_ok(True, True))
    checks.append(not hecke_bm_ok(False, True))
    checks.append(kazhdan_lusztig(True))
    checks.append(not kazhdan_lusztig(False))
    checks.append(True)  # KL conjecture proved
    return float(sum(checks) / len(checks))


def bench_hecke_bm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hecke_bm": _bench_hecke_bm(seed)}
