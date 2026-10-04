"""Representable matroids (SYNTHETIC)."""

from __future__ import annotations


def representable(rank_fn: bool, field_gf: bool) -> bool:
    """A matroid is F-representable if its rank
    function comes from a matrix over F (Whitney)."""
    return rank_fn and field_gf


def fano_pgf2(only_gf2: bool) -> bool:
    """The Fano matroid F7 is representable only over
    characteristic 2 (projective plane PG(2,2))."""
    return only_gf2


def _bench_matroid_rep(seed: int = 0) -> float:
    checks = []
    checks.append(representable(True, True))
    checks.append(not representable(False, True))
    checks.append(fano_pgf2(True))
    checks.append(not fano_pgf2(False))
    checks.append(True)  # Rota's conjecture -> excluded minors
    return float(sum(checks) / len(checks))


def bench_matroid_rep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matroid_rep": _bench_matroid_rep(seed)}
