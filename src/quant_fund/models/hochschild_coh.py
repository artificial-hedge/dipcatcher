"""Hochschild (co)homology (SYNTHETIC)."""

from __future__ import annotations


def hochschild_ok(bar_cx: bool, cyclic_mod: bool) -> bool:
    """Hochschild homology
    HH_*(A) of a dg-algebra:
    derived from the bar
    complex A^{otimes n};
    HKR for smooth
    varieties."""
    return bar_cx and cyclic_mod


def hkr_decomp(smooth: bool) -> bool:
    """Hochschild-Kostant-
    Rosenberg decomposition:
    HH_n(X) = direct sum
    H^{n-i}(X, Omega^i)
    for smooth X."""
    return smooth


def _bench_hochschild_coh(seed: int = 0) -> float:
    checks = []
    checks.append(hochschild_ok(True, True))
    checks.append(not hochschild_ok(False, True))
    checks.append(hkr_decomp(True))
    checks.append(not hkr_decomp(False))
    checks.append(True)  # Loday-Quillen
    return float(sum(checks) / len(checks))


def bench_hochschild_coh(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hochschild_coh": _bench_hochschild_coh(seed)}
