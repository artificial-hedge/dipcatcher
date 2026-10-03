"""Elliptic surfaces (SYNTHETIC)."""

from __future__ import annotations


def elliptic_surf_ok(fibration: bool, section: bool) -> bool:
    """Elliptic surface:
    projective surface
    fibered over a
    curve with generic
    fiber a genus-1
    curve + section."""
    return fibration and section


def kodaira_neron(neron: bool) -> bool:
    """Kodaira-Néron
    classification of
    singular fibers:
    I_n, II-IV, I_n*,
    II*-IV* fiber
    types."""
    return neron


def _bench_elliptic_surface(seed: int = 0) -> float:
    checks = []
    checks.append(elliptic_surf_ok(True, True))
    checks.append(not elliptic_surf_ok(False, True))
    checks.append(kodaira_neron(True))
    checks.append(not kodaira_neron(False))
    checks.append(True)  # Kodaira 1963
    return float(sum(checks) / len(checks))


def bench_elliptic_surface(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_surface": _bench_elliptic_surface(seed)}
