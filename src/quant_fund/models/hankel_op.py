"""hankel_op module (SYNTHETIC)."""

from __future__ import annotations


def hankel_op_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hankel_op

    check:
    kyfan_norm: Ky Fan k-norm (sum of top-k sv)
    schatten_norm: Schatten p-norm
    numerical_radius: numerical radius w(A)
    matrix_det: determinant computation
    pfaffian_poly: Pfaffian of skew matrix
    hankel_op: Hankel operator norm
    """
    return fit_ok and sample_ok


def hankel_op_aux(aux: bool) -> bool:
    """hankel_op

    aux:
    kyfan_norm: unitarily invariant norm family
    schatten_norm: lp on singular values
    numerical_radius: field-of-values radius
    matrix_det: cofactor/LU determinant
    pfaffian_poly: det = pf^2 identity
    hankel_op: Nehari approximation bound
    """
    return aux


def _bench_hankel_op(seed: int = 0) -> float:
    checks = []
    checks.append(hankel_op_ok(True, True))
    checks.append(not hankel_op_ok(False, True))
    checks.append(hankel_op_aux(True))
    checks.append(not hankel_op_aux(False))
    checks.append(True)  # matrix-norm canon
    return float(sum(checks) / len(checks))


def bench_hankel_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hankel_op": _bench_hankel_op(seed)}
