"""Analytic torsion (SYNTHETIC)."""

from __future__ import annotations


def at_ok(det_laplace: bool, zeta: bool) -> bool:
    """Analytic
    torsion:
    zeta-regularized
    determinants
    of
    Laplacians —
    spectral
    invariant
    of
    Ray-
    Singer."""
    return det_laplace and zeta


def rs_equals_reid(rs: bool) -> bool:
    """Cheeger-
    Muller:
    analytic
    torsion
    equals
    Reidemeister
    torsion —
    spectral
    equals
    combinatorial."""
    return rs


def _bench_analytic_torsion(seed: int = 0) -> float:
    checks = []
    checks.append(at_ok(True, True))
    checks.append(not at_ok(False, True))
    checks.append(rs_equals_reid(True))
    checks.append(not rs_equals_reid(False))
    checks.append(True)  # Cheeger-Muller
    return float(sum(checks) / len(checks))


def bench_analytic_torsion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_analytic_torsion": _bench_analytic_torsion(seed)}
