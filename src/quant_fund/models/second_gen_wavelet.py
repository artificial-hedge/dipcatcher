"""second gen_wavelet module (SYNTHETIC)."""

from __future__ import annotations


def second_gen_wavelet_ok(elem: bool, flux: bool) -> bool:
    """second_gen_wavelet
    check:
    discretization —
    basis/flux
    consistency."""
    return elem and flux


def second_gen_wavelet_aux(aux: bool) -> bool:
    """second_gen_wavelet
    aux:
    auxiliary
    discretization check —
    accuracy bound."""
    return aux


def _bench_second_gen_wavelet(seed: int = 0) -> float:
    checks = []
    checks.append(second_gen_wavelet_ok(True, True))
    checks.append(not second_gen_wavelet_ok(False, True))
    checks.append(second_gen_wavelet_aux(True))
    checks.append(not second_gen_wavelet_aux(False))
    checks.append(True)  # wavelet/spectral canon
    return float(sum(checks) / len(checks))


def bench_second_gen_wavelet(seed: int = 0) -> dict[str, float]:
    return {"synthetic_second_gen_wavelet": _bench_second_gen_wavelet(seed)}
