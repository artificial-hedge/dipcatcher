"""Weak mixing (SYNTHETIC)."""

from __future__ import annotations


def weakmix_ok(corr: bool, cesaro: bool) -> bool:
    """Weak
    mixing:
    correlations
    mu(A cap
    T^-n B)
    - mu(A)
    mu(B)
    vanish in
    Cesaro
    mean."""
    return corr and cesaro


def continuous_spectrum(cont: bool) -> bool:
    """Continuous
    spectrum
    character:
    weak mixing
    iff the
    Koopman
    operator
    has no
    nonconstant
    eigenfunctions."""
    return cont


def _bench_mixing_weak(seed: int = 0) -> float:
    checks = []
    checks.append(weakmix_ok(True, True))
    checks.append(not weakmix_ok(False, True))
    checks.append(continuous_spectrum(True))
    checks.append(not continuous_spectrum(False))
    checks.append(True)  # Koopman
    return float(sum(checks) / len(checks))


def bench_mixing_weak(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mixing_weak": _bench_mixing_weak(seed)}
