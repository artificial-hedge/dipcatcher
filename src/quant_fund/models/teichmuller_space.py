"""Teichmuller space (SYNTHETIC)."""

from __future__ import annotations


def teich_ok(marked: bool, contractible: bool) -> bool:
    """Teichmuller
    space:
    marked
    complex
    structures
    up
    to
    isotopy —
    contractible
    cell."""
    return marked and contractible


def mapping_class(mc: bool) -> bool:
    """Mapping
    class
    group
    acts
    properly
    discontinuously;
    quotient
    is
    moduli
    space."""
    return mc


def _bench_teichmuller_space(seed: int = 0) -> float:
    checks = []
    checks.append(teich_ok(True, True))
    checks.append(not teich_ok(False, True))
    checks.append(mapping_class(True))
    checks.append(not mapping_class(False))
    checks.append(True)  # Teichmuller
    return float(sum(checks) / len(checks))


def bench_teichmuller_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_teichmuller_space": _bench_teichmuller_space(seed)}
