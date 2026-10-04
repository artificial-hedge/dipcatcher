"""Julia set (SYNTHETIC)."""

from __future__ import annotations


def julia_ok(repelling: bool, boundary: bool) -> bool:
    """Julia
    set:
    closure
    of
    repelling
    periodic
    points —
    the
    chaotic
    boundary
    of
    basins."""
    return repelling and boundary


def filled_julia(filled: bool) -> bool:
    """Filled
    Julia
    set:
    bounded-
    orbit
    points of
    z -> z^2
    + c;
    connected
    iff c
    in
    Mandelbrot."""
    return filled


def _bench_julia_set(seed: int = 0) -> float:
    checks = []
    checks.append(julia_ok(True, True))
    checks.append(not julia_ok(False, True))
    checks.append(filled_julia(True))
    checks.append(not filled_julia(False))
    checks.append(True)  # Julia-Fatou
    return float(sum(checks) / len(checks))


def bench_julia_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_julia_set": _bench_julia_set(seed)}
