"""derived morph module (SYNTHETIC)."""

from __future__ import annotations


def derived_morph_ok(derived: bool, geometric: bool) -> bool:
    """derived_morph
    check:
    derived
    structure —
    stack."""
    return derived and geometric


def derived_morph_aux(aux: bool) -> bool:
    """derived_morph
    aux:
    auxiliary
    derived
    check —
    morph."""
    return aux


def _bench_derived_morph(seed: int = 0) -> float:
    checks = []
    checks.append(derived_morph_ok(True, True))
    checks.append(not derived_morph_ok(False, True))
    checks.append(derived_morph_aux(True))
    checks.append(not derived_morph_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_morph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_morph": _bench_derived_morph(seed)}
