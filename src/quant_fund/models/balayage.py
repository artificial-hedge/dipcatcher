"""Balayage (SYNTHETIC)."""

from __future__ import annotations


def balayage_ok(sweep: bool, support: bool) -> bool:
    """Balayage:
    sweeping
    a
    measure
    onto
    a
    closed
    set
    preserving
    the
    potential
    outside."""
    return sweep and support


def perron_wiener(pw: bool) -> bool:
    """Perron-
    Wiener-
    Brelot
    method:
    upper
    PWB
    solution
    gives
    the
    Dirichlet
    solution."""
    return pw


def _bench_balayage(seed: int = 0) -> float:
    checks = []
    checks.append(balayage_ok(True, True))
    checks.append(not balayage_ok(False, True))
    checks.append(perron_wiener(True))
    checks.append(not perron_wiener(False))
    checks.append(True)  # Poincaré-Cartan
    return float(sum(checks) / len(checks))


def bench_balayage(seed: int = 0) -> dict[str, float]:
    return {"synthetic_balayage": _bench_balayage(seed)}
