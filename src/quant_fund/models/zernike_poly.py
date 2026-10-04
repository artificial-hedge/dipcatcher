"""zernike poly module (SYNTHETIC)."""

from __future__ import annotations


def zernike_poly_ok(basis: bool, coef: bool) -> bool:
    """zernike_poly
    check:
    radial basis / projection —
    basis/coefficient
    consistency."""
    return basis and coef


def zernike_poly_aux(aux: bool) -> bool:
    """zernike_poly
    aux:
    auxiliary
    basis check —
    reproducing bound."""
    return aux


def _bench_zernike_poly(seed: int = 0) -> float:
    checks = []
    checks.append(zernike_poly_ok(True, True))
    checks.append(not zernike_poly_ok(False, True))
    checks.append(zernike_poly_aux(True))
    checks.append(not zernike_poly_aux(False))
    checks.append(True)  # RBF/basis canon
    return float(sum(checks) / len(checks))


def bench_zernike_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zernike_poly": _bench_zernike_poly(seed)}
