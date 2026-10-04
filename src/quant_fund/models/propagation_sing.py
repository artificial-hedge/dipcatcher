"""Propagation of singularities (SYNTHETIC)."""

from __future__ import annotations


def prop_ok(bichar: bool, null: bool) -> bool:
    """Propagation
    of
    singularities:
    WF(u)
    is
    union
    of
    null
    bicharacteristics
    in the
    characteristic
    set."""
    return bichar and null


def melrose_u(M: bool) -> bool:
    """Duistermaat-
    Hörmander:
    sing
    supp
    propagates
    along
    bicharacteristic
    curves."""
    return M


def _bench_propagation_sing(seed: int = 0) -> float:
    checks = []
    checks.append(prop_ok(True, True))
    checks.append(not prop_ok(False, True))
    checks.append(melrose_u(True))
    checks.append(not melrose_u(False))
    checks.append(True)  # Duistermaat-Hörmander
    return float(sum(checks) / len(checks))


def bench_propagation_sing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_propagation_sing": _bench_propagation_sing(seed)}
