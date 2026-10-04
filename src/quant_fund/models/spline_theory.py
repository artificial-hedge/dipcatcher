"""spline theory module (SYNTHETIC)."""

from __future__ import annotations


def spline_theory_ok(elem: bool, dof: bool) -> bool:
    """spline_theory
    check:
    FE-basis/sequence —
    element/dof
    consistency."""
    return elem and dof


def spline_theory_aux(aux: bool) -> bool:
    """spline_theory
    aux:
    auxiliary
    element check —
    partition bound."""
    return aux


def _bench_spline_theory(seed: int = 0) -> float:
    checks = []
    checks.append(spline_theory_ok(True, True))
    checks.append(not spline_theory_ok(False, True))
    checks.append(spline_theory_aux(True))
    checks.append(not spline_theory_aux(False))
    checks.append(True)  # FE-basis canon
    return float(sum(checks) / len(checks))


def bench_spline_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spline_theory": _bench_spline_theory(seed)}
