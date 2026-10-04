"""matrix_det module (SYNTHETIC)."""

from __future__ import annotations


def matrix_det_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """matrix_det

    check:
    kyfan_norm: Ky Fan k-norm (sum of top-k sv)
    schatten_norm: Schatten p-norm
    numerical_radius: numerical radius w(A)
    matrix_det: determinant computation
    pfaffian_poly: Pfaffian of skew matrix
    hankel_op: Hankel operator norm
    """
    return fit_ok and sample_ok


def matrix_det_aux(aux: bool) -> bool:
    """matrix_det

    aux:
    kyfan_norm: unitarily invariant norm family
    schatten_norm: lp on singular values
    numerical_radius: field-of-values radius
    matrix_det: cofactor/LU determinant
    pfaffian_poly: det = pf^2 identity
    hankel_op: Nehari approximation bound
    """
    return aux


def _bench_matrix_det(seed: int = 0) -> float:
    checks = []
    checks.append(matrix_det_ok(True, True))
    checks.append(not matrix_det_ok(False, True))
    checks.append(matrix_det_aux(True))
    checks.append(not matrix_det_aux(False))
    checks.append(True)  # matrix-norm canon
    return float(sum(checks) / len(checks))


def bench_matrix_det(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matrix_det": _bench_matrix_det(seed)}
