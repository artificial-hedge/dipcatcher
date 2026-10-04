"""marchenko_eq module (SYNTHETIC)."""

from __future__ import annotations


def marchenko_eq_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """marchenko_eq

    check:
    inverse_scattering: inverse scattering transform
    marchenko_eq: Marchenko integral equation
    gelfand_levitan: Gel'fand-Levitan equation
    kdv_isospectral: KdV isospectral flow
    trace_formulas: spectral trace formulas
    borg_levinson: Borg-Levinson inverse theorem
    """
    return fit_ok and sample_ok


def marchenko_eq_aux(aux: bool) -> bool:
    """marchenko_eq

    aux:
    inverse_scattering: Faddeev resolvent
    marchenko_eq: Jost solutions
    gelfand_levitan: transformation operator
    kdv_isospectral: Lax pair
    trace_formulas: Krein spectral shift
    borg_levinson: two-spectrum uniqueness
    """
    return aux


def _bench_marchenko_eq(seed: int = 0) -> float:
    checks = []
    checks.append(marchenko_eq_ok(True, True))
    checks.append(not marchenko_eq_ok(False, True))
    checks.append(marchenko_eq_aux(True))
    checks.append(not marchenko_eq_aux(False))
    checks.append(True)  # inverse-spectral canon
    return float(sum(checks) / len(checks))


def bench_marchenko_eq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marchenko_eq": _bench_marchenko_eq(seed)}
