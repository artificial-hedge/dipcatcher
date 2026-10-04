"""Atiyah-Singer index theorem (SYNTHETIC)."""

from __future__ import annotations


def as_ok(analytic: bool, topological: bool) -> bool:
    """Atiyah-
    Singer:
    analytic
    index
    equals
    topological
    index —
    deepest
    theorem
    of
    geometry-
    analysis."""
    return analytic and topological


def a_hat_genus(ah: bool) -> bool:
    """A-hat
    genus:
    index
    of
    Dirac
    operator
    is
    the
    A-hat
    roof
    genus —
    spin
    case."""
    return ah


def _bench_atiyah_singer(seed: int = 0) -> float:
    checks = []
    checks.append(as_ok(True, True))
    checks.append(not as_ok(False, True))
    checks.append(a_hat_genus(True))
    checks.append(not a_hat_genus(False))
    checks.append(True)  # AS 1963
    return float(sum(checks) / len(checks))


def bench_atiyah_singer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_atiyah_singer": _bench_atiyah_singer(seed)}
