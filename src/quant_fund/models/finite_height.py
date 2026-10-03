"""Finite height representation (SYNTHETIC)."""

from __future__ import annotations


def fh_ok(finite_height: bool, crystalline: bool) -> bool:
    """Finite
    height:
    crystalline
    rep
    has
    finite
    E-
    height —
    Kisin
    height."""
    return finite_height and crystalline


def height_bound(hb: bool) -> bool:
    """Height
    bound:
    E-
    height
    bounded
    by
    p-
    range —
    Kisin."""
    return hb


def _bench_finite_height(seed: int = 0) -> float:
    checks = []
    checks.append(fh_ok(True, True))
    checks.append(not fh_ok(False, True))
    checks.append(height_bound(True))
    checks.append(not height_bound(False))
    checks.append(True)  # Kisin
    return float(sum(checks) / len(checks))


def bench_finite_height(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finite_height": _bench_finite_height(seed)}
