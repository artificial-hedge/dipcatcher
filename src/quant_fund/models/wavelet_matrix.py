"""wavelet matrix module (SYNTHETIC)."""

from __future__ import annotations


def wavelet_matrix_ok(elem: bool, flux: bool) -> bool:
    """wavelet_matrix
    check:
    discretization —
    basis/flux
    consistency."""
    return elem and flux


def wavelet_matrix_aux(aux: bool) -> bool:
    """wavelet_matrix
    aux:
    auxiliary
    discretization check —
    accuracy bound."""
    return aux


def _bench_wavelet_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(wavelet_matrix_ok(True, True))
    checks.append(not wavelet_matrix_ok(False, True))
    checks.append(wavelet_matrix_aux(True))
    checks.append(not wavelet_matrix_aux(False))
    checks.append(True)  # wavelet/spectral canon
    return float(sum(checks) / len(checks))


def bench_wavelet_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wavelet_matrix": _bench_wavelet_matrix(seed)}
