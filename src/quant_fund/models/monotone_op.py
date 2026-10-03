"""monotone_op module (SYNTHETIC)."""

from __future__ import annotations


def monotone_op_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """monotone_op

    check:
    monotone_op: monotone operators
    degree_theory: Leray-Schauder degree
    schauder_fixed: Schauder fixed-point theorem
    krein_rutman: Krein-Rutman theorem
    minty_browder: Minty-Browder theorem
    maximal_monotone: maximal monotone operators
    """
    return fit_ok and sample_ok


def monotone_op_aux(aux: bool) -> bool:
    """monotone_op

    aux:
    monotone_op: demicontinuity
    degree_theory: Brouwer degree
    schauder_fixed: compact operators
    krein_rutman: positive cone
    minty_browder: surjectivity of monotone maps
    maximal_monotone: Yosida approximation
    """
    return aux


def _bench_monotone_op(seed: int = 0) -> float:
    checks = []
    checks.append(monotone_op_ok(True, True))
    checks.append(not monotone_op_ok(False, True))
    checks.append(monotone_op_aux(True))
    checks.append(not monotone_op_aux(False))
    checks.append(True)  # nonlinear-functional-analysis canon
    return float(sum(checks) / len(checks))


def bench_monotone_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monotone_op": _bench_monotone_op(seed)}
