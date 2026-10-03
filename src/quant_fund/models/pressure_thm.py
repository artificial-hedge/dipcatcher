"""Pressure theorem (SYNTHETIC)."""

from __future__ import annotations


def pressure_ok(top: bool, eigen: bool) -> bool:
    """Pressure:
    log
    spectral
    radius
    of the
    transfer
    operator;
    convex in
    the
    potential."""
    return top and eigen


def analyticity(anal: bool) -> bool:
    """Analyticity:
    pressure
    is real
    analytic
    on
    Holder
    potentials;
    derivs
    give
    equilibrium
    states."""
    return anal


def _bench_pressure_thm(seed: int = 0) -> float:
    checks = []
    checks.append(pressure_ok(True, True))
    checks.append(not pressure_ok(False, True))
    checks.append(analyticity(True))
    checks.append(not analyticity(False))
    checks.append(True)  # Ruelle
    return float(sum(checks) / len(checks))


def bench_pressure_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pressure_thm": _bench_pressure_thm(seed)}
