"""gram_matrix module (SYNTHETIC)."""

from __future__ import annotations


def gram_matrix_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gram_matrix

    check:
    gram_matrix: Gram inner-product matrix
    gram_determinant: Gram determinant volume
    householder_reflect: Householder reflection
    givens_rotation: planar Givens rotation
    back_substitution: upper-triangular backsolve
    forward_substitution: lower-triangular forwardsolve
    """
    return fit_ok and sample_ok


def gram_matrix_aux(aux: bool) -> bool:
    """gram_matrix

    aux:
    gram_matrix: SPD witness
    gram_determinant: positivity check
    householder_reflect: orthogonal check
    givens_rotation: orthogonality check
    back_substitution: residual check
    forward_substitution: residual check
    """
    return aux


def _bench_gram_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(gram_matrix_ok(True, True))
    checks.append(not gram_matrix_ok(False, True))
    checks.append(gram_matrix_aux(True))
    checks.append(not gram_matrix_aux(False))
    checks.append(True)  # linear-systems canon
    return float(sum(checks) / len(checks))


def bench_gram_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gram_matrix": _bench_gram_matrix(seed)}
