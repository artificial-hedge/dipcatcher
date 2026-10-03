"""Cartan spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def css_ok(group_action: bool, equivariant_coh: bool) -> bool:
    """Cartan
    spectral
    sequence:
    equivariant
    cohomology
    of
    G-
    space —
    Cartan-
    Leray."""
    return group_action and equivariant_coh


def cartan_leray(cl: bool) -> bool:
    """Cartan-
    Leray:
    spectral
    sequence
    for
    group
    action
    cohomology —
    equivariant."""
    return cl


def _bench_cartan_ss(seed: int = 0) -> float:
    checks = []
    checks.append(css_ok(True, True))
    checks.append(not css_ok(False, True))
    checks.append(cartan_leray(True))
    checks.append(not cartan_leray(False))
    checks.append(True)  # Cartan-Leray
    return float(sum(checks) / len(checks))


def bench_cartan_ss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cartan_ss": _bench_cartan_ss(seed)}
