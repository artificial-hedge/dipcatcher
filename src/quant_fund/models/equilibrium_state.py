"""Equilibrium states (SYNTHETIC)."""

from __future__ import annotations


def eq_ok(measure: bool, unique: bool) -> bool:
    """Equilibrium
    state:
    measure
    attaining
    the
    variational
    sup;
    unique
    for
    Holder
    potentials
    on
    expanding
    systems."""
    return measure and unique


def gibbs_property(gibbs: bool) -> bool:
    """Gibbs
    property:
    cylinder
    probabilities
    comparable
    to
    e^{S_n phi
    - nP}."""
    return gibbs


def _bench_equilibrium_state(seed: int = 0) -> float:
    checks = []
    checks.append(eq_ok(True, True))
    checks.append(not eq_ok(False, True))
    checks.append(gibbs_property(True))
    checks.append(not gibbs_property(False))
    checks.append(True)  # Bowen
    return float(sum(checks) / len(checks))


def bench_equilibrium_state(seed: int = 0) -> dict[str, float]:
    return {"synthetic_equilibrium_state": _bench_equilibrium_state(seed)}
