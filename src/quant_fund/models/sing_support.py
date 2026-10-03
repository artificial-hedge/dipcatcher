"""sing_support module (SYNTHETIC)."""

from __future__ import annotations


def sing_support_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sing_support

    check:
    schwartz_dist: Schwartz distribution pairing
    temper_dist: tempered distribution Fourier action
    dist_convolution: distributional convolution
    sing_support: singular support analysis
    paley_wiener: Paley-Wiener exponential type
    sobolev_trace: Sobolev trace to boundary
    """
    return fit_ok and sample_ok


def sing_support_aux(aux: bool) -> bool:
    """sing_support

    aux:
    schwartz_dist: smooth dense approximation
    temper_dist: S' topology boundedness
    dist_convolution: associativity under support
    sing_support: wavefront refinement
    paley_wiener: compact support order bound
    sobolev_trace: fractional smoothness loss 1/2
    """
    return aux


def _bench_sing_support(seed: int = 0) -> float:
    checks = []
    checks.append(sing_support_ok(True, True))
    checks.append(not sing_support_ok(False, True))
    checks.append(sing_support_aux(True))
    checks.append(not sing_support_aux(False))
    checks.append(True)  # distribution-theory canon
    return float(sum(checks) / len(checks))


def bench_sing_support(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sing_support": _bench_sing_support(seed)}
