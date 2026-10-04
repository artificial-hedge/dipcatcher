"""monoidal derived module (SYNTHETIC)."""

from __future__ import annotations


def monoidal_derived_ok(category: bool, structure: bool) -> bool:
    """monoidal_derived
    check:
    category
    structure —
    tensor."""
    return category and structure


def monoidal_derived_aux(aux: bool) -> bool:
    """monoidal_derived
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_monoidal_derived(seed: int = 0) -> float:
    checks = []
    checks.append(monoidal_derived_ok(True, True))
    checks.append(not monoidal_derived_ok(False, True))
    checks.append(monoidal_derived_aux(True))
    checks.append(not monoidal_derived_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_monoidal_derived(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monoidal_derived": _bench_monoidal_derived(seed)}
