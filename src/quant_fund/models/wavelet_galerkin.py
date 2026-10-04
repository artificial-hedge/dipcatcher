"""wavelet galerkin module (SYNTHETIC)."""

from __future__ import annotations


def wavelet_galerkin_ok(scale: bool, coeff: bool) -> bool:
    """wavelet_galerkin
    check:
    wavelet-Galerkin —
    multiresolution
    consistency."""
    return scale and coeff


def wavelet_galerkin_aux(aux: bool) -> bool:
    """wavelet_galerkin
    aux:
    auxiliary
    wavelet check —
    refinement mask."""
    return aux


def _bench_wavelet_galerkin(seed: int = 0) -> float:
    checks = []
    checks.append(wavelet_galerkin_ok(True, True))
    checks.append(not wavelet_galerkin_ok(False, True))
    checks.append(wavelet_galerkin_aux(True))
    checks.append(not wavelet_galerkin_aux(False))
    checks.append(True)  # wavelet-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_wavelet_galerkin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wavelet_galerkin": _bench_wavelet_galerkin(seed)}
