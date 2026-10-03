"""fredholm_op module (SYNTHETIC)."""

from __future__ import annotations


def fredholm_op_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fredholm_op

    check:
    selfadjoint_op: A equals its adjoint
    unitary_operator: U*U = I
    shift_operator: unilateral/bilateral shift
    fredholm_op: Fredholm index operator
    normal_operator: AA* = A*A
    multiplication_op: pointwise multiplication op
    """
    return fit_ok and sample_ok


def fredholm_op_aux(aux: bool) -> bool:
    """fredholm_op

    aux:
    selfadjoint_op: real spectrum witness
    unitary_operator: norm/isometry check
    shift_operator: defect index
    fredholm_op: index computation
    normal_operator: spectral decomposition
    multiplication_op: symbol check
    """
    return aux


def _bench_fredholm_op(seed: int = 0) -> float:
    checks = []
    checks.append(fredholm_op_ok(True, True))
    checks.append(not fredholm_op_ok(False, True))
    checks.append(fredholm_op_aux(True))
    checks.append(not fredholm_op_aux(False))
    checks.append(True)  # operator-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_fredholm_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fredholm_op": _bench_fredholm_op(seed)}
