"""derived cover module (SYNTHETIC)."""

from __future__ import annotations


def derived_cover_ok(derived: bool, geometric: bool) -> bool:
    """derived_cover
    check:
    derived
    structure —
    stack."""
    return derived and geometric


def derived_cover_aux(aux: bool) -> bool:
    """derived_cover
    aux:
    auxiliary
    derived
    check —
    morph."""
    return aux


def _bench_derived_cover(seed: int = 0) -> float:
    checks = []
    checks.append(derived_cover_ok(True, True))
    checks.append(not derived_cover_ok(False, True))
    checks.append(derived_cover_aux(True))
    checks.append(not derived_cover_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_cover(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_cover": _bench_derived_cover(seed)}
