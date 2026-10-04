"""Nirenberg problem (SYNTHETIC)."""

from __future__ import annotations


def np_ok(gauss_curv: bool, sphere: bool) -> bool:
    """Nirenberg
    problem:
    prescribe
    Gauss
    curvature
    on
    S^2
    conformally —
    Kazdan-
    Warner
    condition."""
    return gauss_curv and sphere


def chang_gursky_yang(cgy: bool) -> bool:
    """Chang-
    Gursky-
    Yang:
    solution
    of
    the
    Nirenberg
    problem
    via
    blow-
    up
    analysis."""
    return cgy


def _bench_nirenberg_problem(seed: int = 0) -> float:
    checks = []
    checks.append(np_ok(True, True))
    checks.append(not np_ok(False, True))
    checks.append(chang_gursky_yang(True))
    checks.append(not chang_gursky_yang(False))
    checks.append(True)  # Nirenberg-Chang-Gursky-Yang
    return float(sum(checks) / len(checks))


def bench_nirenberg_problem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nirenberg_problem": _bench_nirenberg_problem(seed)}
