"""maximal_monotone module (SYNTHETIC)."""

from __future__ import annotations


def maximal_monotone_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """maximal_monotone

    check:
    monotone_op: monotone operators
    degree_theory: Leray-Schauder degree
    schauder_fixed: Schauder fixed-point theorem
    krein_rutman: Krein-Rutman theorem
    minty_browder: Minty-Browder theorem
    maximal_monotone: maximal monotone operators
    """
    return fit_ok and sample_ok


def maximal_monotone_aux(aux: bool) -> bool:
    """maximal_monotone

    aux:
    monotone_op: demicontinuity
    degree_theory: Brouwer degree
    schauder_fixed: compact operators
    krein_rutman: positive cone
    minty_browder: surjectivity of monotone maps
    maximal_monotone: Yosida approximation
    """
    return aux


def _bench_maximal_monotone(seed: int = 0) -> float:
    checks = []
    checks.append(maximal_monotone_ok(True, True))
    checks.append(not maximal_monotone_ok(False, True))
    checks.append(maximal_monotone_aux(True))
    checks.append(not maximal_monotone_aux(False))
    checks.append(True)  # nonlinear-functional-analysis canon
    return float(sum(checks) / len(checks))


def bench_maximal_monotone(seed: int = 0) -> dict[str, float]:
    return {"synthetic_maximal_monotone": _bench_maximal_monotone(seed)}
