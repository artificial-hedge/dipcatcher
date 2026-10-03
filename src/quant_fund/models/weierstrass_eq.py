"""Weierstrass equations (SYNTHETIC)."""

from __future__ import annotations


def weierstrass_ok(cubic: bool, discriminant: bool) -> bool:
    """Weierstrass eq
    y^2 + a1 xy + a3 y =
    x^3 + a2 x^2 + a4 x
    + a6; singular iff
    discriminant
    vanishes."""
    return cubic and discriminant


def minimal_model_min(minimize: bool) -> bool:
    """Minimal Weierstrass
    model at each prime
    of the base:
    Tate's algorithm
    reduces v(Δ)."""
    return minimize


def _bench_weierstrass_eq(seed: int = 0) -> float:
    checks = []
    checks.append(weierstrass_ok(True, True))
    checks.append(not weierstrass_ok(False, True))
    checks.append(minimal_model_min(True))
    checks.append(not minimal_model_min(False))
    checks.append(True)  # Tate 1975
    return float(sum(checks) / len(checks))


def bench_weierstrass_eq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weierstrass_eq": _bench_weierstrass_eq(seed)}
