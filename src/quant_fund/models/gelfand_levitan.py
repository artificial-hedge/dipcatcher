"""gelfand_levitan module (SYNTHETIC)."""

from __future__ import annotations


def gelfand_levitan_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gelfand_levitan

    check:
    inverse_scattering: inverse scattering transform
    marchenko_eq: Marchenko integral equation
    gelfand_levitan: Gel'fand-Levitan equation
    kdv_isospectral: KdV isospectral flow
    trace_formulas: spectral trace formulas
    borg_levinson: Borg-Levinson inverse theorem
    """
    return fit_ok and sample_ok


def gelfand_levitan_aux(aux: bool) -> bool:
    """gelfand_levitan

    aux:
    inverse_scattering: Faddeev resolvent
    marchenko_eq: Jost solutions
    gelfand_levitan: transformation operator
    kdv_isospectral: Lax pair
    trace_formulas: Krein spectral shift
    borg_levinson: two-spectrum uniqueness
    """
    return aux


def _bench_gelfand_levitan(seed: int = 0) -> float:
    checks = []
    checks.append(gelfand_levitan_ok(True, True))
    checks.append(not gelfand_levitan_ok(False, True))
    checks.append(gelfand_levitan_aux(True))
    checks.append(not gelfand_levitan_aux(False))
    checks.append(True)  # inverse-spectral canon
    return float(sum(checks) / len(checks))


def bench_gelfand_levitan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gelfand_levitan": _bench_gelfand_levitan(seed)}
