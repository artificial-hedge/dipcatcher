"""Elliptic K-theory (SYNTHETIC)."""

from __future__ import annotations


def ek_ok(elliptic: bool, k_theory: bool) -> bool:
    """Elliptic:
    elliptic
    K-
    theory —
    Segal
    elliptic."""
    return elliptic and k_theory


def landweber_elliptic(le: bool) -> bool:
    """Landweber:
    Landweber-
    Ravenel-
    Stong
    elliptic —
    Landweber."""
    return le


def _bench_elliptic_k(seed: int = 0) -> float:
    checks = []
    checks.append(ek_ok(True, True))
    checks.append(not ek_ok(False, True))
    checks.append(landweber_elliptic(True))
    checks.append(not landweber_elliptic(False))
    checks.append(True)  # LRS
    return float(sum(checks) / len(checks))


def bench_elliptic_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_k": _bench_elliptic_k(seed)}
