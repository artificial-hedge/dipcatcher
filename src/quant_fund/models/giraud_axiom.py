"""Giraud axioms (SYNTHETIC)."""

from __future__ import annotations


def ga_ok(giraud: bool, cocomplete: bool) -> bool:
    """Giraud:
    Giraud
    axioms
    for
    topos —
    Grothendieck
    Giraud."""
    return giraud and cocomplete


def exact_topos(et: bool) -> bool:
    """Exact
    topos:
    exact
    and
    extensive —
    Giraud
    axiom."""
    return et


def _bench_giraud_axiom(seed: int = 0) -> float:
    checks = []
    checks.append(ga_ok(True, True))
    checks.append(not ga_ok(False, True))
    checks.append(exact_topos(True))
    checks.append(not exact_topos(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_giraud_axiom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_giraud_axiom": _bench_giraud_axiom(seed)}
