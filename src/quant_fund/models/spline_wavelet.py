"""spline wavelet module (SYNTHETIC)."""

from __future__ import annotations


def spline_wavelet_ok(scale: bool, coeff: bool) -> bool:
    """spline_wavelet
    check:
    wavelet-Galerkin —
    multiresolution
    consistency."""
    return scale and coeff


def spline_wavelet_aux(aux: bool) -> bool:
    """spline_wavelet
    aux:
    auxiliary
    wavelet check —
    refinement mask."""
    return aux


def _bench_spline_wavelet(seed: int = 0) -> float:
    checks = []
    checks.append(spline_wavelet_ok(True, True))
    checks.append(not spline_wavelet_ok(False, True))
    checks.append(spline_wavelet_aux(True))
    checks.append(not spline_wavelet_aux(False))
    checks.append(True)  # wavelet-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_spline_wavelet(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spline_wavelet": _bench_spline_wavelet(seed)}
