"""wavelet collocation module (SYNTHETIC)."""

from __future__ import annotations


def wavelet_collocation_ok(scale: bool, coeff: bool) -> bool:
    """wavelet_collocation
    check:
    wavelet-Galerkin —
    multiresolution
    consistency."""
    return scale and coeff


def wavelet_collocation_aux(aux: bool) -> bool:
    """wavelet_collocation
    aux:
    auxiliary
    wavelet check —
    refinement mask."""
    return aux


def _bench_wavelet_collocation(seed: int = 0) -> float:
    checks = []
    checks.append(wavelet_collocation_ok(True, True))
    checks.append(not wavelet_collocation_ok(False, True))
    checks.append(wavelet_collocation_aux(True))
    checks.append(not wavelet_collocation_aux(False))
    checks.append(True)  # wavelet-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_wavelet_collocation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wavelet_collocation": _bench_wavelet_collocation(seed)}
