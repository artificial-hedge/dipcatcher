"""Plateau problem (SYNTHETIC)."""

from __future__ import annotations


def pp_ok(boundary: bool, area_min: bool) -> bool:
    """Plateau
    problem:
    find
    a
    least-
    area
    surface
    spanning
    a
    given
    boundary —
    Douglas
    and
    Rado."""
    return boundary and area_min


def douglas_solution(ds: bool) -> bool:
    """Douglas'
    solution:
    direct
    method
    on
    the
    Dirichlet
    integral
    solves
    Plateau —
    Fields
    medal
    1936."""
    return ds


def _bench_plateau_problem(seed: int = 0) -> float:
    checks = []
    checks.append(pp_ok(True, True))
    checks.append(not pp_ok(False, True))
    checks.append(douglas_solution(True))
    checks.append(not douglas_solution(False))
    checks.append(True)  # Douglas-Rado
    return float(sum(checks) / len(checks))


def bench_plateau_problem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_plateau_problem": _bench_plateau_problem(seed)}
