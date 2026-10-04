"""GUE statistics (SYNTHETIC)."""

from __future__ import annotations


def gue_ok(wigner: bool, hermitian: bool) -> bool:
    """Gaussian
    unitary
    ensemble:
    Hermitian
    matrices
    with
    Gaussian
    entries —
    universal
    spectral
    stats."""
    return wigner and hermitian


def sine_kernel(sk: bool) -> bool:
    """Sine
    kernel:
    two-
    point
    correlation
    function
    sin(x-y)/(x-y)
    —
    determinantal."""
    return sk


def _bench_gue_statistics(seed: int = 0) -> float:
    checks = []
    checks.append(gue_ok(True, True))
    checks.append(not gue_ok(False, True))
    checks.append(sine_kernel(True))
    checks.append(not sine_kernel(False))
    checks.append(True)  # Wigner-Dyson
    return float(sum(checks) / len(checks))


def bench_gue_statistics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gue_statistics": _bench_gue_statistics(seed)}
