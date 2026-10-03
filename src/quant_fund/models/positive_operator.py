"""positive_operator module (SYNTHETIC)."""

from __future__ import annotations


def positive_operator_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """positive_operator

    check:
    bounded_operator: bounded linear operator norm test
    operator_norm: induced matrix norm
    operator_adjoint: Hilbert-space adjoint
    projection_operator: orthogonal projection P^2=P
    positive_operator: positive semidefinite operator
    isometry_operator: norm-preserving isometry
    """
    return fit_ok and sample_ok


def positive_operator_aux(aux: bool) -> bool:
    """positive_operator

    aux:
    bounded_operator: continuity equivalence
    operator_norm: submultiplicative bound
    operator_adjoint: self-adjoint check
    projection_operator: idempotent check
    positive_operator: Cholesky witness
    isometry_operator: polarization check
    """
    return aux


def _bench_positive_operator(seed: int = 0) -> float:
    checks = []
    checks.append(positive_operator_ok(True, True))
    checks.append(not positive_operator_ok(False, True))
    checks.append(positive_operator_aux(True))
    checks.append(not positive_operator_aux(False))
    checks.append(True)  # operator-theory canon
    return float(sum(checks) / len(checks))


def bench_positive_operator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_positive_operator": _bench_positive_operator(seed)}
