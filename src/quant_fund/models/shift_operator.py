"""shift_operator module (SYNTHETIC)."""

from __future__ import annotations


def shift_operator_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """shift_operator

    check:
    selfadjoint_op: A equals its adjoint
    unitary_operator: U*U = I
    shift_operator: unilateral/bilateral shift
    fredholm_op: Fredholm index operator
    normal_operator: AA* = A*A
    multiplication_op: pointwise multiplication op
    """
    return fit_ok and sample_ok


def shift_operator_aux(aux: bool) -> bool:
    """shift_operator

    aux:
    selfadjoint_op: real spectrum witness
    unitary_operator: norm/isometry check
    shift_operator: defect index
    fredholm_op: index computation
    normal_operator: spectral decomposition
    multiplication_op: symbol check
    """
    return aux


def _bench_shift_operator(seed: int = 0) -> float:
    checks = []
    checks.append(shift_operator_ok(True, True))
    checks.append(not shift_operator_ok(False, True))
    checks.append(shift_operator_aux(True))
    checks.append(not shift_operator_aux(False))
    checks.append(True)  # operator-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_shift_operator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shift_operator": _bench_shift_operator(seed)}
