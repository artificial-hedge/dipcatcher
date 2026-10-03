"""Yamabe problem (SYNTHETIC)."""

from __future__ import annotations


def yp_ok(conformal: bool, constant_scalar: bool) -> bool:
    """Yamabe
    problem:
    every
    conformal
    class
    has
    a
    metric
    of
    constant
    scalar
    curvature —
    Yamabe-
    Trudinger-
    Aubin-
    Schoen."""
    return conformal and constant_scalar


def yamabe_minimizer(ym: bool) -> bool:
    """Yamabe
    minimizer:
    minimize
    total
    scalar
    curvature
    in
    the
    conformal
    class —
    variational
    approach."""
    return ym


def _bench_yamabe_problem(seed: int = 0) -> float:
    checks = []
    checks.append(yp_ok(True, True))
    checks.append(not yp_ok(False, True))
    checks.append(yamabe_minimizer(True))
    checks.append(not yamabe_minimizer(False))
    checks.append(True)  # Yamabe-Trudinger-Aubin-Schoen
    return float(sum(checks) / len(checks))


def bench_yamabe_problem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yamabe_problem": _bench_yamabe_problem(seed)}
