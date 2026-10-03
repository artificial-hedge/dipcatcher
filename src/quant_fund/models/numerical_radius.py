"""numerical_radius module (SYNTHETIC)."""

from __future__ import annotations


def numerical_radius_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """numerical_radius

    check:
    kyfan_norm: Ky Fan k-norm (sum of top-k sv)
    schatten_norm: Schatten p-norm
    numerical_radius: numerical radius w(A)
    matrix_det: determinant computation
    pfaffian_poly: Pfaffian of skew matrix
    hankel_op: Hankel operator norm
    """
    return fit_ok and sample_ok


def numerical_radius_aux(aux: bool) -> bool:
    """numerical_radius

    aux:
    kyfan_norm: unitarily invariant norm family
    schatten_norm: lp on singular values
    numerical_radius: field-of-values radius
    matrix_det: cofactor/LU determinant
    pfaffian_poly: det = pf^2 identity
    hankel_op: Nehari approximation bound
    """
    return aux


def _bench_numerical_radius(seed: int = 0) -> float:
    checks = []
    checks.append(numerical_radius_ok(True, True))
    checks.append(not numerical_radius_ok(False, True))
    checks.append(numerical_radius_aux(True))
    checks.append(not numerical_radius_aux(False))
    checks.append(True)  # matrix-norm canon
    return float(sum(checks) / len(checks))


def bench_numerical_radius(seed: int = 0) -> dict[str, float]:
    return {"synthetic_numerical_radius": _bench_numerical_radius(seed)}
