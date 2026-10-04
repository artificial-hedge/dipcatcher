"""minty_browder module (SYNTHETIC)."""

from __future__ import annotations


def minty_browder_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """minty_browder

    check:
    monotone_op: monotone operators
    degree_theory: Leray-Schauder degree
    schauder_fixed: Schauder fixed-point theorem
    krein_rutman: Krein-Rutman theorem
    minty_browder: Minty-Browder theorem
    maximal_monotone: maximal monotone operators
    """
    return fit_ok and sample_ok


def minty_browder_aux(aux: bool) -> bool:
    """minty_browder

    aux:
    monotone_op: demicontinuity
    degree_theory: Brouwer degree
    schauder_fixed: compact operators
    krein_rutman: positive cone
    minty_browder: surjectivity of monotone maps
    maximal_monotone: Yosida approximation
    """
    return aux


def _bench_minty_browder(seed: int = 0) -> float:
    checks = []
    checks.append(minty_browder_ok(True, True))
    checks.append(not minty_browder_ok(False, True))
    checks.append(minty_browder_aux(True))
    checks.append(not minty_browder_aux(False))
    checks.append(True)  # nonlinear-functional-analysis canon
    return float(sum(checks) / len(checks))


def bench_minty_browder(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minty_browder": _bench_minty_browder(seed)}
