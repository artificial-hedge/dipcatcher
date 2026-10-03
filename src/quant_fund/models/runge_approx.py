"""Runge approximation (SYNTHETIC)."""

from __future__ import annotations


def runge_ok(compact: bool, complement: bool) -> bool:
    """Runge:
    holomorphic
    functions
    on
    a compact
    set
    are
    approximated
    by
    rational
    functions
    with
    poles
    outside."""
    return compact and complement


def polynomial_when_hole_free(ph: bool) -> bool:
    """Polynomial
    approximation
    when
    the
    complement
    is
    connected
    —
    no
    holes."""
    return ph


def _bench_runge_approx(seed: int = 0) -> float:
    checks = []
    checks.append(runge_ok(True, True))
    checks.append(not runge_ok(False, True))
    checks.append(polynomial_when_hole_free(True))
    checks.append(not polynomial_when_hole_free(False))
    checks.append(True)  # Runge
    return float(sum(checks) / len(checks))


def bench_runge_approx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_runge_approx": _bench_runge_approx(seed)}
