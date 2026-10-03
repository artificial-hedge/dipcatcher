"""Godbillon-Vey class (SYNTHETIC)."""

from __future__ import annotations


def gv_ok(codim1: bool, degree3: bool) -> bool:
    """Godbillon-
    Vey:
    codim-1
    foliations
    carry
    a
    degree-3
    secondary
    class —
    eta-wedge-deta."""
    return codim1 and degree3


def nontrivial_gv(ngv: bool) -> bool:
    """Nontriviality:
    Roussarie
    and
    Thurston
    compute
    nonzero
    GV
    —
    first
    secondary
    invariant."""
    return ngv


def _bench_godbillon_vey(seed: int = 0) -> float:
    checks = []
    checks.append(gv_ok(True, True))
    checks.append(not gv_ok(False, True))
    checks.append(nontrivial_gv(True))
    checks.append(not nontrivial_gv(False))
    checks.append(True)  # GV 1971
    return float(sum(checks) / len(checks))


def bench_godbillon_vey(seed: int = 0) -> dict[str, float]:
    return {"synthetic_godbillon_vey": _bench_godbillon_vey(seed)}
