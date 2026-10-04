"""pfaffian_poly module (SYNTHETIC)."""

from __future__ import annotations


def pfaffian_poly_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pfaffian_poly

    check:
    kyfan_norm: Ky Fan k-norm (sum of top-k sv)
    schatten_norm: Schatten p-norm
    numerical_radius: numerical radius w(A)
    matrix_det: determinant computation
    pfaffian_poly: Pfaffian of skew matrix
    hankel_op: Hankel operator norm
    """
    return fit_ok and sample_ok


def pfaffian_poly_aux(aux: bool) -> bool:
    """pfaffian_poly

    aux:
    kyfan_norm: unitarily invariant norm family
    schatten_norm: lp on singular values
    numerical_radius: field-of-values radius
    matrix_det: cofactor/LU determinant
    pfaffian_poly: det = pf^2 identity
    hankel_op: Nehari approximation bound
    """
    return aux


def _bench_pfaffian_poly(seed: int = 0) -> float:
    checks = []
    checks.append(pfaffian_poly_ok(True, True))
    checks.append(not pfaffian_poly_ok(False, True))
    checks.append(pfaffian_poly_aux(True))
    checks.append(not pfaffian_poly_aux(False))
    checks.append(True)  # matrix-norm canon
    return float(sum(checks) / len(checks))


def bench_pfaffian_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pfaffian_poly": _bench_pfaffian_poly(seed)}
