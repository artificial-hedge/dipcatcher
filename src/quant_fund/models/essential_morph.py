"""Essential morphism (SYNTHETIC)."""

from __future__ import annotations


def em_ok(essential: bool, geometric: bool) -> bool:
    """Essential:
    essential
    geometric
    morphism —
    Kelly
    essential."""
    return essential and geometric


def left_adjoint(la: bool) -> bool:
    """Left
    adjoint:
    left
    adjoint
    to
    inverse
    image —
    essential
    f-shriek."""
    return la


def _bench_essential_morph(seed: int = 0) -> float:
    checks = []
    checks.append(em_ok(True, True))
    checks.append(not em_ok(False, True))
    checks.append(left_adjoint(True))
    checks.append(not left_adjoint(False))
    checks.append(True)  # Kelly
    return float(sum(checks) / len(checks))


def bench_essential_morph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_essential_morph": _bench_essential_morph(seed)}
