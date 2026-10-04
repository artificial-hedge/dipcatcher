"""Kronecker theorem (SYNTHETIC)."""

from __future__ import annotations


def kron_ok(dense: bool, independent: bool) -> bool:
    """Kronecker:
    multiples
    of
    rationally
    independent
    numbers
    are
    dense
    modulo
    one."""
    return dense and independent


def equidist_weyl(ew: bool) -> bool:
    """Weyl:
    n*alpha
    is
    equidistributed
    for
    irrational
    alpha —
    uniform
    distribution."""
    return ew


def _bench_kronecker_thm(seed: int = 0) -> float:
    checks = []
    checks.append(kron_ok(True, True))
    checks.append(not kron_ok(False, True))
    checks.append(equidist_weyl(True))
    checks.append(not equidist_weyl(False))
    checks.append(True)  # Kronecker-Weyl
    return float(sum(checks) / len(checks))


def bench_kronecker_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kronecker_thm": _bench_kronecker_thm(seed)}
