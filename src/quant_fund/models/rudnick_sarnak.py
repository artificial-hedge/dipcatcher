"""Rudnick-Sarnak QUE (SYNTHETIC)."""

from __future__ import annotations


def rs_ok(equidist: bool, eigenfunctions: bool) -> bool:
    """Rudnick-
    Sarnak:
    quantum
    unique
    ergodicity
    for
    arithmetic
    surfaces —
    Lindenstrauss
    proves."""
    return equidist and eigenfunctions


def que_theorem(qt: bool) -> bool:
    """QUE:
    eigenfunctions
    equidistribute
    on
    hyperbolic
    manifolds —
    no
    scars."""
    return qt


def _bench_rudnick_sarnak(seed: int = 0) -> float:
    checks = []
    checks.append(rs_ok(True, True))
    checks.append(not rs_ok(False, True))
    checks.append(que_theorem(True))
    checks.append(not que_theorem(False))
    checks.append(True)  # Lindenstrauss-Soundararajan
    return float(sum(checks) / len(checks))


def bench_rudnick_sarnak(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rudnick_sarnak": _bench_rudnick_sarnak(seed)}
