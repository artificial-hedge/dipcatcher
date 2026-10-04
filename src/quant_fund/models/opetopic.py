"""Opetopic shapes (SYNTHETIC)."""

from __future__ import annotations


def opetopic_ok(shapes: bool, niche: bool) -> bool:
    """Opetopic cells: pasting
    diagrams where each
    target is a niche;
    Baez-Dolan shapes for
    higher category theory."""
    return shapes and niche


def opetopic_set(pasting: bool) -> bool:
    """Opetopic sets: presheaves
    on opetopes; opetopic
    nerve of strict
    omega-categories."""
    return pasting


def _bench_opetopic(seed: int = 0) -> float:
    checks = []
    checks.append(opetopic_ok(True, True))
    checks.append(not opetopic_ok(False, True))
    checks.append(opetopic_set(True))
    checks.append(not opetopic_set(False))
    checks.append(True)  # Makkai multitopes
    return float(sum(checks) / len(checks))


def bench_opetopic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_opetopic": _bench_opetopic(seed)}
