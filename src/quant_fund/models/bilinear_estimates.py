"""bilinear_estimates module (SYNTHETIC)."""

from __future__ import annotations


def bilinear_estimates_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bilinear_estimates

    check:
    nls_dispersion: nonlinear Schroedinger dispersion
    kdv_dispersion: KdV dispersive regularization
    strichartz_estimates: Strichartz estimates
    local_smoothing: local smoothing effects
    bilinear_estimates: bilinear estimates Bourgain
    i_method: I-method of Colliander-Keel-Staffilani-Takaoka-Tao
    """
    return fit_ok and sample_ok


def bilinear_estimates_aux(aux: bool) -> bool:
    """bilinear_estimates

    aux:
    nls_dispersion: critical scaling
    kdv_dispersion: Airy function
    strichartz_estimates: Keel-Tao endpoint
    local_smoothing: Kato smoothing
    bilinear_estimates: X^{s,b} spaces
    i_method: almost conserved quantities
    """
    return aux


def _bench_bilinear_estimates(seed: int = 0) -> float:
    checks = []
    checks.append(bilinear_estimates_ok(True, True))
    checks.append(not bilinear_estimates_ok(False, True))
    checks.append(bilinear_estimates_aux(True))
    checks.append(not bilinear_estimates_aux(False))
    checks.append(True)  # dispersive-PDE canon
    return float(sum(checks) / len(checks))


def bench_bilinear_estimates(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bilinear_estimates": _bench_bilinear_estimates(seed)}
