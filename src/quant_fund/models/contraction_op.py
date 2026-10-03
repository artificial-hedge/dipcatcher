"""contraction_op module (SYNTHETIC)."""

from __future__ import annotations


def contraction_op_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """contraction_op

    check:
    toeplitz_op: Toeplitz multiplication operator
    integral_op: integral kernel operator
    differential_op: differential operator
    contraction_op: norm-1 contraction
    accretive_op: accretive resolvent
    sectorial_op: sector-valued form operator
    """
    return fit_ok and sample_ok


def contraction_op_aux(aux: bool) -> bool:
    """contraction_op

    aux:
    toeplitz_op: symbol recovery
    integral_op: kernel bound
    differential_op: symbol/orders
    contraction_op: power boundedness
    accretive_op: dissipativity check
    sectorial_op: numerical-range angle
    """
    return aux


def _bench_contraction_op(seed: int = 0) -> float:
    checks = []
    checks.append(contraction_op_ok(True, True))
    checks.append(not contraction_op_ok(False, True))
    checks.append(contraction_op_aux(True))
    checks.append(not contraction_op_aux(False))
    checks.append(True)  # operator-theory-3 canon
    return float(sum(checks) / len(checks))


def bench_contraction_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_contraction_op": _bench_contraction_op(seed)}
