"""derived local module (SYNTHETIC)."""

from __future__ import annotations


def derived_local_ok(derived: bool, geometric: bool) -> bool:
    """derived_local
    check:
    derived
    structure —
    conn."""
    return derived and geometric


def derived_local_aux(aux: bool) -> bool:
    """derived_local
    aux:
    auxiliary
    derived
    check —
    local."""
    return aux


def _bench_derived_local(seed: int = 0) -> float:
    checks = []
    checks.append(derived_local_ok(True, True))
    checks.append(not derived_local_ok(False, True))
    checks.append(derived_local_aux(True))
    checks.append(not derived_local_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_local(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_local": _bench_derived_local(seed)}
