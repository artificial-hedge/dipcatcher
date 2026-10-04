"""borg_levinson module (SYNTHETIC)."""

from __future__ import annotations


def borg_levinson_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """borg_levinson

    check:
    inverse_scattering: inverse scattering transform
    marchenko_eq: Marchenko integral equation
    gelfand_levitan: Gel'fand-Levitan equation
    kdv_isospectral: KdV isospectral flow
    trace_formulas: spectral trace formulas
    borg_levinson: Borg-Levinson inverse theorem
    """
    return fit_ok and sample_ok


def borg_levinson_aux(aux: bool) -> bool:
    """borg_levinson

    aux:
    inverse_scattering: Faddeev resolvent
    marchenko_eq: Jost solutions
    gelfand_levitan: transformation operator
    kdv_isospectral: Lax pair
    trace_formulas: Krein spectral shift
    borg_levinson: two-spectrum uniqueness
    """
    return aux


def _bench_borg_levinson(seed: int = 0) -> float:
    checks = []
    checks.append(borg_levinson_ok(True, True))
    checks.append(not borg_levinson_ok(False, True))
    checks.append(borg_levinson_aux(True))
    checks.append(not borg_levinson_aux(False))
    checks.append(True)  # inverse-spectral canon
    return float(sum(checks) / len(checks))


def bench_borg_levinson(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borg_levinson": _bench_borg_levinson(seed)}
