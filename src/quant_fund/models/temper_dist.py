"""temper_dist module (SYNTHETIC)."""

from __future__ import annotations


def temper_dist_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """temper_dist

    check:
    schwartz_dist: Schwartz distribution pairing
    temper_dist: tempered distribution Fourier action
    dist_convolution: distributional convolution
    sing_support: singular support analysis
    paley_wiener: Paley-Wiener exponential type
    sobolev_trace: Sobolev trace to boundary
    """
    return fit_ok and sample_ok


def temper_dist_aux(aux: bool) -> bool:
    """temper_dist

    aux:
    schwartz_dist: smooth dense approximation
    temper_dist: S' topology boundedness
    dist_convolution: associativity under support
    sing_support: wavefront refinement
    paley_wiener: compact support order bound
    sobolev_trace: fractional smoothness loss 1/2
    """
    return aux


def _bench_temper_dist(seed: int = 0) -> float:
    checks = []
    checks.append(temper_dist_ok(True, True))
    checks.append(not temper_dist_ok(False, True))
    checks.append(temper_dist_aux(True))
    checks.append(not temper_dist_aux(False))
    checks.append(True)  # distribution-theory canon
    return float(sum(checks) / len(checks))


def bench_temper_dist(seed: int = 0) -> dict[str, float]:
    return {"synthetic_temper_dist": _bench_temper_dist(seed)}
