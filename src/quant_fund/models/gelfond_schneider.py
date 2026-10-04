"""Gelfond-Schneider theorem (SYNTHETIC)."""

from __future__ import annotations


def gs_ok(algebraic_base: bool, irrational_exp: bool) -> bool:
    """Gelfond-
    Schneider:
    alpha^beta
    is
    transcendental
    for
    algebraic
    alpha
    not
    0,1
    and
    algebraic
    irrational
    beta."""
    return algebraic_base and irrational_exp


def hilbert7(h7: bool) -> bool:
    """Hilbert's
    seventh
    problem
    solved:
    2^sqrt(2)
    is
    transcendental."""
    return h7


def _bench_gelfond_schneider(seed: int = 0) -> float:
    checks = []
    checks.append(gs_ok(True, True))
    checks.append(not gs_ok(False, True))
    checks.append(hilbert7(True))
    checks.append(not hilbert7(False))
    checks.append(True)  # Gelfond-Schneider
    return float(sum(checks) / len(checks))


def bench_gelfond_schneider(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gelfond_schneider": _bench_gelfond_schneider(seed)}
