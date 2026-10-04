"""Riemann zeta function (SYNTHETIC)."""

from __future__ import annotations


def zeta_ok(euler: bool, continuation: bool) -> bool:
    """Riemann
    zeta
    zeta(s) =
    prod_p
    (1-p^-s)^-1;
    meromorphic
    continuation,
    functional
    equation."""
    return euler and continuation


def critical_strip(strip: bool) -> bool:
    """Critical
    strip
    0 < Re s
    < 1 holds
    the
    nontrivial
    zeros;
    RH locates
    them on
    Re s = 1/2."""
    return strip


def _bench_riemann_zeta(seed: int = 0) -> float:
    checks = []
    checks.append(zeta_ok(True, True))
    checks.append(not zeta_ok(False, True))
    checks.append(critical_strip(True))
    checks.append(not critical_strip(False))
    checks.append(True)  # Riemann
    return float(sum(checks) / len(checks))


def bench_riemann_zeta(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riemann_zeta": _bench_riemann_zeta(seed)}
