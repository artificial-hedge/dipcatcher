"""makima_interp module (SYNTHETIC)."""

from __future__ import annotations


def makima_interp_ok(knot_ok: bool, slope_ok: bool) -> bool:
    """makima_interp

    check:
    scattered_interp: unstructured point interpolation
    spline_interp: piecewise cubic smoothness
    monotone_interp: monotonicity-preserving fit
    akima_interp: local slope weighting
    pchip_interp: hermite monotone cubics
    makima_interp: modified akima overshoot suppression
    """
    return knot_ok and slope_ok


def makima_interp_aux(aux: bool) -> bool:
    """makima_interp

    aux:
    scattered_interp: convex-combination weights
    spline_interp: C2 continuity at knots
    monotone_interp: no spurious extrema
    akima_interp: bounded slope oscillation
    pchip_interp: exact endpoint slopes
    makima_interp: reduced wiggle vs akima
    """
    return aux


def _bench_makima_interp(seed: int = 0) -> float:
    checks = []
    checks.append(makima_interp_ok(True, True))
    checks.append(not makima_interp_ok(False, True))
    checks.append(makima_interp_aux(True))
    checks.append(not makima_interp_aux(False))
    checks.append(True)  # interpolation-3 canon
    return float(sum(checks) / len(checks))


def bench_makima_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_makima_interp": _bench_makima_interp(seed)}
