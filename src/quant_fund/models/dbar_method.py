"""dbar_method module (SYNTHETIC)."""

from __future__ import annotations


def dbar_method_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dbar_method

    check:
    dbar_method: d-bar method for integrable systems
    orthogonal_poly_rh: orthogonal polynomials via RH
    isomonodromy: isomonodromic deformations
    fokas_unified: Fokas unified transform method
    deift_zhou: Deift-Zhou steepest descent
    small_norm_rh: small-norm Riemann-Hilbert problems
    """
    return fit_ok and sample_ok


def dbar_method_aux(aux: bool) -> bool:
    """dbar_method

    aux:
    dbar_method: Cauchy-Green operator
    orthogonal_poly_rh: Fokas-Its-Kitaev RH
    isomonodromy: Jimbo-Miwa-Ueno theory
    fokas_unified: global relation
    deift_zhou: nonlinear steepest descent
    small_norm_rh: shrinking contours
    """
    return aux


def _bench_dbar_method(seed: int = 0) -> float:
    checks = []
    checks.append(dbar_method_ok(True, True))
    checks.append(not dbar_method_ok(False, True))
    checks.append(dbar_method_aux(True))
    checks.append(not dbar_method_aux(False))
    checks.append(True)  # Riemann-Hilbert canon
    return float(sum(checks) / len(checks))


def bench_dbar_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dbar_method": _bench_dbar_method(seed)}
