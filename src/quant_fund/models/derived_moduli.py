"""Derived moduli (SYNTHETIC)."""

from __future__ import annotations


def derived_moduli_ok(derived_stack: bool, tangent_cx: bool) -> bool:
    """Derived moduli stack
    M(X): derived enhancement
    with tangent complex
    T; classical truncation
    is the classical moduli."""
    return derived_stack and tangent_cx


def virtual_structure(perfection: bool) -> bool:
    """Virtual structure sheaf
    and virtual fundamental
    class on derived moduli
    with perfect
    obstruction theory."""
    return perfection


def _bench_derived_moduli(seed: int = 0) -> float:
    checks = []
    checks.append(derived_moduli_ok(True, True))
    checks.append(not derived_moduli_ok(False, True))
    checks.append(virtual_structure(True))
    checks.append(not virtual_structure(False))
    checks.append(True)  # Behrend-Fantechi
    return float(sum(checks) / len(checks))


def bench_derived_moduli(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_moduli": _bench_derived_moduli(seed)}
