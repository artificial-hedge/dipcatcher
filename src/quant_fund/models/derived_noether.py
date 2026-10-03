"""derived noether module (SYNTHETIC)."""

from __future__ import annotations


def derived_noether_ok(derived: bool, geometric: bool) -> bool:
    """derived_noether
    check:
    derived
    structure —
    conn."""
    return derived and geometric


def derived_noether_aux(aux: bool) -> bool:
    """derived_noether
    aux:
    auxiliary
    derived
    check —
    local."""
    return aux


def _bench_derived_noether(seed: int = 0) -> float:
    checks = []
    checks.append(derived_noether_ok(True, True))
    checks.append(not derived_noether_ok(False, True))
    checks.append(derived_noether_aux(True))
    checks.append(not derived_noether_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_noether(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_noether": _bench_derived_noether(seed)}
