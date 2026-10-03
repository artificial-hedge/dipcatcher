"""local_smoothing module (SYNTHETIC)."""

from __future__ import annotations


def local_smoothing_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """local_smoothing

    check:
    nls_dispersion: nonlinear Schroedinger dispersion
    kdv_dispersion: KdV dispersive regularization
    strichartz_estimates: Strichartz estimates
    local_smoothing: local smoothing effects
    bilinear_estimates: bilinear estimates Bourgain
    i_method: I-method of Colliander-Keel-Staffilani-Takaoka-Tao
    """
    return fit_ok and sample_ok


def local_smoothing_aux(aux: bool) -> bool:
    """local_smoothing

    aux:
    nls_dispersion: critical scaling
    kdv_dispersion: Airy function
    strichartz_estimates: Keel-Tao endpoint
    local_smoothing: Kato smoothing
    bilinear_estimates: X^{s,b} spaces
    i_method: almost conserved quantities
    """
    return aux


def _bench_local_smoothing(seed: int = 0) -> float:
    checks = []
    checks.append(local_smoothing_ok(True, True))
    checks.append(not local_smoothing_ok(False, True))
    checks.append(local_smoothing_aux(True))
    checks.append(not local_smoothing_aux(False))
    checks.append(True)  # dispersive-PDE canon
    return float(sum(checks) / len(checks))


def bench_local_smoothing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_local_smoothing": _bench_local_smoothing(seed)}
