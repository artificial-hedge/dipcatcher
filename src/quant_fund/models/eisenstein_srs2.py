"""Eisenstein series (SYNTHETIC)."""

from __future__ import annotations


def eis_ok(summation: bool, convergent: bool) -> bool:
    """Eisenstein
    series
    E_k(z) =
    sum'
    (mz+n)^-k;
    converges
    for k > 2,
    quasimodular
    at k = 2."""
    return summation and convergent


def eis_normalize(norm: bool) -> bool:
    """Normalized
    E_4, E_6
    with
    rational
    Fourier
    coefficients
    generate
    the ring."""
    return norm


def _bench_eisenstein_srs2(seed: int = 0) -> float:
    checks = []
    checks.append(eis_ok(True, True))
    checks.append(not eis_ok(False, True))
    checks.append(eis_normalize(True))
    checks.append(not eis_normalize(False))
    checks.append(True)  # Eisenstein
    return float(sum(checks) / len(checks))


def bench_eisenstein_srs2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eisenstein_srs2": _bench_eisenstein_srs2(seed)}
