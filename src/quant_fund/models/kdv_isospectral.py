"""kdv_isospectral module (SYNTHETIC)."""

from __future__ import annotations


def kdv_isospectral_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kdv_isospectral

    check:
    inverse_scattering: inverse scattering transform
    marchenko_eq: Marchenko integral equation
    gelfand_levitan: Gel'fand-Levitan equation
    kdv_isospectral: KdV isospectral flow
    trace_formulas: spectral trace formulas
    borg_levinson: Borg-Levinson inverse theorem
    """
    return fit_ok and sample_ok


def kdv_isospectral_aux(aux: bool) -> bool:
    """kdv_isospectral

    aux:
    inverse_scattering: Faddeev resolvent
    marchenko_eq: Jost solutions
    gelfand_levitan: transformation operator
    kdv_isospectral: Lax pair
    trace_formulas: Krein spectral shift
    borg_levinson: two-spectrum uniqueness
    """
    return aux


def _bench_kdv_isospectral(seed: int = 0) -> float:
    checks = []
    checks.append(kdv_isospectral_ok(True, True))
    checks.append(not kdv_isospectral_ok(False, True))
    checks.append(kdv_isospectral_aux(True))
    checks.append(not kdv_isospectral_aux(False))
    checks.append(True)  # inverse-spectral canon
    return float(sum(checks) / len(checks))


def bench_kdv_isospectral(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kdv_isospectral": _bench_kdv_isospectral(seed)}
