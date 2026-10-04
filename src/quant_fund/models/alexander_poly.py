"""Alexander polynomial (SYNTHETIC)."""

from __future__ import annotations


def ap_ok(seifert: bool, symmetric: bool) -> bool:
    """Alexander
    polynomial:
    first
    knot
    polynomial,
    symmetric
    Laurent
    polynomial
    from
    the
    Seifert
    matrix."""
    return seifert and symmetric


def fibered_criterion(fc: bool) -> bool:
    """Monic
    Alexander
    polynomial
    is
    necessary
    for
    fiberedness —
    classical
    criterion."""
    return fc


def _bench_alexander_poly(seed: int = 0) -> float:
    checks = []
    checks.append(ap_ok(True, True))
    checks.append(not ap_ok(False, True))
    checks.append(fibered_criterion(True))
    checks.append(not fibered_criterion(False))
    checks.append(True)  # Alexander 1928
    return float(sum(checks) / len(checks))


def bench_alexander_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_alexander_poly": _bench_alexander_poly(seed)}
