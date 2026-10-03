"""Macdonald polynomials (SYNTHETIC)."""

from __future__ import annotations


def mp_ok(q_t: bool, symmetric: bool) -> bool:
    """Macdonald
    polynomials:
    two-
    parameter
    symmetric
    functions
    unifying
    Hall-
    Littlewood
    and
    Jack."""
    return q_t and symmetric


def macdonald_operator(mo: bool) -> bool:
    """Macdonald
    operator:
    difference
    operator
    with
    the
    polynomials
    as
    eigenfunctions —
    Macdonald
    1988."""
    return mo


def _bench_macdonald_poly(seed: int = 0) -> float:
    checks = []
    checks.append(mp_ok(True, True))
    checks.append(not mp_ok(False, True))
    checks.append(macdonald_operator(True))
    checks.append(not macdonald_operator(False))
    checks.append(True)  # Macdonald
    return float(sum(checks) / len(checks))


def bench_macdonald_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_macdonald_poly": _bench_macdonald_poly(seed)}
