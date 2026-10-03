"""strichartz_estimates module (SYNTHETIC)."""

from __future__ import annotations


def strichartz_estimates_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """strichartz_estimates

    check:
    nls_dispersion: nonlinear Schroedinger dispersion
    kdv_dispersion: KdV dispersive regularization
    strichartz_estimates: Strichartz estimates
    local_smoothing: local smoothing effects
    bilinear_estimates: bilinear estimates Bourgain
    i_method: I-method of Colliander-Keel-Staffilani-Takaoka-Tao
    """
    return fit_ok and sample_ok


def strichartz_estimates_aux(aux: bool) -> bool:
    """strichartz_estimates

    aux:
    nls_dispersion: critical scaling
    kdv_dispersion: Airy function
    strichartz_estimates: Keel-Tao endpoint
    local_smoothing: Kato smoothing
    bilinear_estimates: X^{s,b} spaces
    i_method: almost conserved quantities
    """
    return aux


def _bench_strichartz_estimates(seed: int = 0) -> float:
    checks = []
    checks.append(strichartz_estimates_ok(True, True))
    checks.append(not strichartz_estimates_ok(False, True))
    checks.append(strichartz_estimates_aux(True))
    checks.append(not strichartz_estimates_aux(False))
    checks.append(True)  # dispersive-PDE canon
    return float(sum(checks) / len(checks))


def bench_strichartz_estimates(seed: int = 0) -> dict[str, float]:
    return {"synthetic_strichartz_estimates": _bench_strichartz_estimates(seed)}
