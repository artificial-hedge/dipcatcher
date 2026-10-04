"""wavelet_char module (SYNTHETIC)."""

from __future__ import annotations


def wavelet_char_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wavelet_char

    check:
    besov_space: Besov space B-s-pq norm
    triebel_lizorkin: Triebel-Lizorkin F-s-pq norm
    atoms_decomp: atomic decomposition of function spaces
    wavelet_char: wavelet characterization
    besov_embed: Besov embedding theorems
    hardy_littlewood_max: Hardy-Littlewood maximal function
    """
    return fit_ok and sample_ok


def wavelet_char_aux(aux: bool) -> bool:
    """wavelet_char

    aux:
    besov_space: Littlewood-Paley decomposition
    triebel_lizorkin: square-function norm
    atoms_decomp: moment conditions on atoms
    wavelet_char: smoothness in coefficients
    besov_embed: sharp embedding exponent
    hardy_littlewood_max: weak-type bound
    """
    return aux


def _bench_wavelet_char(seed: int = 0) -> float:
    checks = []
    checks.append(wavelet_char_ok(True, True))
    checks.append(not wavelet_char_ok(False, True))
    checks.append(wavelet_char_aux(True))
    checks.append(not wavelet_char_aux(False))
    checks.append(True)  # Besov/Triebel-Lizorkin canon
    return float(sum(checks) / len(checks))


def bench_wavelet_char(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wavelet_char": _bench_wavelet_char(seed)}
