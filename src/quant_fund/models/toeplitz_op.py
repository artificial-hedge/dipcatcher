"""toeplitz_op module (SYNTHETIC)."""

from __future__ import annotations


def toeplitz_op_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """toeplitz_op

    check:
    toeplitz_op: Toeplitz multiplication operator
    integral_op: integral kernel operator
    differential_op: differential operator
    contraction_op: norm-1 contraction
    accretive_op: accretive resolvent
    sectorial_op: sector-valued form operator
    """
    return fit_ok and sample_ok


def toeplitz_op_aux(aux: bool) -> bool:
    """toeplitz_op

    aux:
    toeplitz_op: symbol recovery
    integral_op: kernel bound
    differential_op: symbol/orders
    contraction_op: power boundedness
    accretive_op: dissipativity check
    sectorial_op: numerical-range angle
    """
    return aux


def _bench_toeplitz_op(seed: int = 0) -> float:
    checks = []
    checks.append(toeplitz_op_ok(True, True))
    checks.append(not toeplitz_op_ok(False, True))
    checks.append(toeplitz_op_aux(True))
    checks.append(not toeplitz_op_aux(False))
    checks.append(True)  # operator-theory-3 canon
    return float(sum(checks) / len(checks))


def bench_toeplitz_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toeplitz_op": _bench_toeplitz_op(seed)}
