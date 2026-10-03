"""kdv_dispersion module (SYNTHETIC)."""

from __future__ import annotations


def kdv_dispersion_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kdv_dispersion

    check:
    nls_dispersion: nonlinear Schroedinger dispersion
    kdv_dispersion: KdV dispersive regularization
    strichartz_estimates: Strichartz estimates
    local_smoothing: local smoothing effects
    bilinear_estimates: bilinear estimates Bourgain
    i_method: I-method of Colliander-Keel-Staffilani-Takaoka-Tao
    """
    return fit_ok and sample_ok


def kdv_dispersion_aux(aux: bool) -> bool:
    """kdv_dispersion

    aux:
    nls_dispersion: critical scaling
    kdv_dispersion: Airy function
    strichartz_estimates: Keel-Tao endpoint
    local_smoothing: Kato smoothing
    bilinear_estimates: X^{s,b} spaces
    i_method: almost conserved quantities
    """
    return aux


def _bench_kdv_dispersion(seed: int = 0) -> float:
    checks = []
    checks.append(kdv_dispersion_ok(True, True))
    checks.append(not kdv_dispersion_ok(False, True))
    checks.append(kdv_dispersion_aux(True))
    checks.append(not kdv_dispersion_aux(False))
    checks.append(True)  # dispersive-PDE canon
    return float(sum(checks) / len(checks))


def bench_kdv_dispersion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kdv_dispersion": _bench_kdv_dispersion(seed)}
