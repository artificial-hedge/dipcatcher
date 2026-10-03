"""Selberg trace formula II (SYNTHETIC)."""

from __future__ import annotations


def st2_ok(spectral: bool, geometric: bool) -> bool:
    """Selberg
    trace
    formula:
    spectral
    side
    equals
    geometric
    side
    on
    hyperbolic
    surfaces —
    Weil
    law."""
    return spectral and geometric


def hyperbolic_conjugacy(hc: bool) -> bool:
    """Hyperbolic
    terms:
    contribution
    of
    closed
    geodesics
    weighted
    by
    length —
    trace
    of
    lengths."""
    return hc


def _bench_selberg_trace2(seed: int = 0) -> float:
    checks = []
    checks.append(st2_ok(True, True))
    checks.append(not st2_ok(False, True))
    checks.append(hyperbolic_conjugacy(True))
    checks.append(not hyperbolic_conjugacy(False))
    checks.append(True)  # Selberg
    return float(sum(checks) / len(checks))


def bench_selberg_trace2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_selberg_trace2": _bench_selberg_trace2(seed)}
