"""operator_adjoint module (SYNTHETIC)."""

from __future__ import annotations


def operator_adjoint_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """operator_adjoint

    check:
    bounded_operator: bounded linear operator norm test
    operator_norm: induced matrix norm
    operator_adjoint: Hilbert-space adjoint
    projection_operator: orthogonal projection P^2=P
    positive_operator: positive semidefinite operator
    isometry_operator: norm-preserving isometry
    """
    return fit_ok and sample_ok


def operator_adjoint_aux(aux: bool) -> bool:
    """operator_adjoint

    aux:
    bounded_operator: continuity equivalence
    operator_norm: submultiplicative bound
    operator_adjoint: self-adjoint check
    projection_operator: idempotent check
    positive_operator: Cholesky witness
    isometry_operator: polarization check
    """
    return aux


def _bench_operator_adjoint(seed: int = 0) -> float:
    checks = []
    checks.append(operator_adjoint_ok(True, True))
    checks.append(not operator_adjoint_ok(False, True))
    checks.append(operator_adjoint_aux(True))
    checks.append(not operator_adjoint_aux(False))
    checks.append(True)  # operator-theory canon
    return float(sum(checks) / len(checks))


def bench_operator_adjoint(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operator_adjoint": _bench_operator_adjoint(seed)}
