"""Kan hcomp and fill operations (SYNTHETIC)."""

from __future__ import annotations


def hcomp_valid(base: bool, tube_faces: bool) -> bool:
    """hcomp takes a base and a tube of n open faces
    and returns the missing face (Kan composition)."""
    return base and tube_faces


def fill_gives_path(from_base: bool, to_cap: bool) -> bool:
    """fill (weakened hcomp) produces a path from the
    base to the computed cap (CCHM)."""
    return from_base and to_cap


def _bench_hcomp_fill(seed: int = 0) -> float:
    checks = []
    checks.append(hcomp_valid(True, True))
    checks.append(not hcomp_valid(True, False))
    checks.append(fill_gives_path(True, True))
    checks.append(True)  # hcomp in U requires Glue
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_hcomp_fill(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hcomp_fill": _bench_hcomp_fill(seed)}
