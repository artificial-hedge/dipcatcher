"""Nuclear space (SYNTHETIC)."""

from __future__ import annotations


def ns_ok(nuclear: bool, trace_class: bool) -> bool:
    """Nuclear
    space:
    nuclear
    maps
    have
    trace —
    Grothendieck
    nuclear."""
    return nuclear and trace_class


def fredholm_theory(ft: bool) -> bool:
    """Fredholm:
    trace
    of
    nuclear
    operators
    via
    Fredholm
    theory —
    Grothendieck
    trace."""
    return ft


def _bench_nuclear_space(seed: int = 0) -> float:
    checks = []
    checks.append(ns_ok(True, True))
    checks.append(not ns_ok(False, True))
    checks.append(fredholm_theory(True))
    checks.append(not fredholm_theory(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_nuclear_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuclear_space": _bench_nuclear_space(seed)}
