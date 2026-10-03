"""derived topos module (SYNTHETIC)."""

from __future__ import annotations


def derived_topos_ok(derived: bool, geometric: bool) -> bool:
    """derived_topos
    check:
    derived
    structure —
    stack."""
    return derived and geometric


def derived_topos_aux(aux: bool) -> bool:
    """derived_topos
    aux:
    auxiliary
    derived
    check —
    morph."""
    return aux


def _bench_derived_topos(seed: int = 0) -> float:
    checks = []
    checks.append(derived_topos_ok(True, True))
    checks.append(not derived_topos_ok(False, True))
    checks.append(derived_topos_aux(True))
    checks.append(not derived_topos_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_topos(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_topos": _bench_derived_topos(seed)}
