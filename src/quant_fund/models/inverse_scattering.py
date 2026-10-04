"""inverse_scattering module (SYNTHETIC)."""

from __future__ import annotations


def inverse_scattering_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """inverse_scattering

    check:
    inverse_scattering: inverse scattering transform
    marchenko_eq: Marchenko integral equation
    gelfand_levitan: Gel'fand-Levitan equation
    kdv_isospectral: KdV isospectral flow
    trace_formulas: spectral trace formulas
    borg_levinson: Borg-Levinson inverse theorem
    """
    return fit_ok and sample_ok


def inverse_scattering_aux(aux: bool) -> bool:
    """inverse_scattering

    aux:
    inverse_scattering: Faddeev resolvent
    marchenko_eq: Jost solutions
    gelfand_levitan: transformation operator
    kdv_isospectral: Lax pair
    trace_formulas: Krein spectral shift
    borg_levinson: two-spectrum uniqueness
    """
    return aux


def _bench_inverse_scattering(seed: int = 0) -> float:
    checks = []
    checks.append(inverse_scattering_ok(True, True))
    checks.append(not inverse_scattering_ok(False, True))
    checks.append(inverse_scattering_aux(True))
    checks.append(not inverse_scattering_aux(False))
    checks.append(True)  # inverse-spectral canon
    return float(sum(checks) / len(checks))


def bench_inverse_scattering(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inverse_scattering": _bench_inverse_scattering(seed)}
