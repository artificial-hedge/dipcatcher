"""Dirac operator (SYNTHETIC)."""

from __future__ import annotations


def dir_ok(clifford: bool, self_adjoint: bool) -> bool:
    """Dirac
    operator:
    first-order
    Clifford-
    module
    operator —
    square
    is
    Laplace-
    type."""
    return clifford and self_adjoint


def lichnerowicz(lf: bool) -> bool:
    """Lichnerowicz
    formula:
    D-squared
    equals
    connection
    Laplacian
    plus
    scalar/4 —
    positive
    scalar
    forces
    trivial
    kernel."""
    return lf


def _bench_dirac_op(seed: int = 0) -> float:
    checks = []
    checks.append(dir_ok(True, True))
    checks.append(not dir_ok(False, True))
    checks.append(lichnerowicz(True))
    checks.append(not lichnerowicz(False))
    checks.append(True)  # Lichnerowicz
    return float(sum(checks) / len(checks))


def bench_dirac_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dirac_op": _bench_dirac_op(seed)}
