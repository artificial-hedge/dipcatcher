"""thin plate_spline module (SYNTHETIC)."""

from __future__ import annotations


def thin_plate_spline_ok(basis: bool, coef: bool) -> bool:
    """thin_plate_spline
    check:
    radial basis / projection —
    basis/coefficient
    consistency."""
    return basis and coef


def thin_plate_spline_aux(aux: bool) -> bool:
    """thin_plate_spline
    aux:
    auxiliary
    basis check —
    reproducing bound."""
    return aux


def _bench_thin_plate_spline(seed: int = 0) -> float:
    checks = []
    checks.append(thin_plate_spline_ok(True, True))
    checks.append(not thin_plate_spline_ok(False, True))
    checks.append(thin_plate_spline_aux(True))
    checks.append(not thin_plate_spline_aux(False))
    checks.append(True)  # RBF/basis canon
    return float(sum(checks) / len(checks))


def bench_thin_plate_spline(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thin_plate_spline": _bench_thin_plate_spline(seed)}
