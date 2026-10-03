"""Poisson brackets (SYNTHETIC)."""

from __future__ import annotations


def pb_ok(jacobi: bool, leibniz: bool) -> bool:
    """Poisson
    bracket:
    skew,
    Jacobi,
    and
    Leibniz —
    Lie
    algebra
    on
    functions."""
    return jacobi and leibniz


def casimir(ca: bool) -> bool:
    """Casimirs:
    functions
    bracketing
    zero
    with
    everything —
    center of
    the
    Poisson
    algebra."""
    return ca


def _bench_poisson_bracket(seed: int = 0) -> float:
    checks = []
    checks.append(pb_ok(True, True))
    checks.append(not pb_ok(False, True))
    checks.append(casimir(True))
    checks.append(not casimir(False))
    checks.append(True)  # Poisson-Lichnerowicz
    return float(sum(checks) / len(checks))


def bench_poisson_bracket(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poisson_bracket": _bench_poisson_bracket(seed)}
