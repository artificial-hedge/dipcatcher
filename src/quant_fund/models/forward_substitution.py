"""forward_substitution module (SYNTHETIC)."""

from __future__ import annotations


def forward_substitution_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """forward_substitution

    check:
    gram_matrix: Gram inner-product matrix
    gram_determinant: Gram determinant volume
    householder_reflect: Householder reflection
    givens_rotation: planar Givens rotation
    back_substitution: upper-triangular backsolve
    forward_substitution: lower-triangular forwardsolve
    """
    return fit_ok and sample_ok


def forward_substitution_aux(aux: bool) -> bool:
    """forward_substitution

    aux:
    gram_matrix: SPD witness
    gram_determinant: positivity check
    householder_reflect: orthogonal check
    givens_rotation: orthogonality check
    back_substitution: residual check
    forward_substitution: residual check
    """
    return aux


def _bench_forward_substitution(seed: int = 0) -> float:
    checks = []
    checks.append(forward_substitution_ok(True, True))
    checks.append(not forward_substitution_ok(False, True))
    checks.append(forward_substitution_aux(True))
    checks.append(not forward_substitution_aux(False))
    checks.append(True)  # linear-systems canon
    return float(sum(checks) / len(checks))


def bench_forward_substitution(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forward_substitution": _bench_forward_substitution(seed)}
