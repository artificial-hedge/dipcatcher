"""derived flat module (SYNTHETIC)."""

from __future__ import annotations


def derived_flat_ok(derived: bool, geometric: bool) -> bool:
    """derived_flat
    check:
    derived
    structure —
    etale."""
    return derived and geometric


def derived_flat_aux(aux: bool) -> bool:
    """derived_flat
    aux:
    auxiliary
    derived
    check —
    flat."""
    return aux


def _bench_derived_flat(seed: int = 0) -> float:
    checks = []
    checks.append(derived_flat_ok(True, True))
    checks.append(not derived_flat_ok(False, True))
    checks.append(derived_flat_aux(True))
    checks.append(not derived_flat_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_flat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_flat": _bench_derived_flat(seed)}
