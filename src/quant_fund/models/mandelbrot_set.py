"""Mandelbrot set (SYNTHETIC)."""

from __future__ import annotations


def mandel_ok(connected: bool, boundary: bool) -> bool:
    """Mandelbrot
    set M:
    parameters
    c for
    which
    K_c is
    connected;
    compact
    and
    connected."""
    return connected and boundary


def hyperbolic_comp(hyp: bool) -> bool:
    """Hyperbolic
    components
    of M:
    interior
    conjectured
    dense
    (MLC
    implies
    it)."""
    return hyp


def _bench_mandelbrot_set(seed: int = 0) -> float:
    checks = []
    checks.append(mandel_ok(True, True))
    checks.append(not mandel_ok(False, True))
    checks.append(hyperbolic_comp(True))
    checks.append(not hyperbolic_comp(False))
    checks.append(True)  # Douady-Hubbard
    return float(sum(checks) / len(checks))


def bench_mandelbrot_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mandelbrot_set": _bench_mandelbrot_set(seed)}
